"""Optional HTTP interface for a single local text embedding model."""

import base64
from threading import Lock
from typing import Annotated, Any, Literal

import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from fastembed.text.text_embedding_base import TextEmbeddingBase

TextInput = Annotated[StrictStr, Field(min_length=1)]


class EmbeddingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: TextInput | Annotated[list[TextInput], Field(min_length=1, max_length=2048)]
    model: StrictStr | None = None
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: Annotated[int, Field(strict=True, gt=0)] | None = None
    user: StrictStr | None = None


def _error(
    message: str,
    status: int = 400,
    param: str | None = None,
    error_type: str = "invalid_request_error",
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": error_type, "param": param, "code": None}},
    )


def create_app(model: TextEmbeddingBase, *, batch_size: int = 256) -> FastAPI:
    """Serve an initialized model. Inference is serialized outside the ASGI event loop."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    app = FastAPI(title="FastEmbed")
    lock = Lock()
    embedding_size = model.embedding_size

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Validation details can contain the original text. Return only the field name.
        location = exc.errors()[0]["loc"]
        param = location[1] if len(location) > 1 and isinstance(location[1], str) else None
        return _error("Invalid embedding request.", param=param)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {"id": model.model_name, "object": "model", "created": 0, "owned_by": "fastembed"}
            ],
        }

    @app.post("/v1/embeddings", response_model=None)
    def embeddings(body: EmbeddingsRequest) -> dict[str, Any] | JSONResponse:
        if body.model is not None and body.model != model.model_name:
            return _error("The requested model is not loaded.", status=404, param="model")
        if body.dimensions is not None and body.dimensions != embedding_size:
            return _error(
                "Only the model's native embedding dimensions are supported.", param="dimensions"
            )
        inputs = [body.input] if isinstance(body.input, str) else body.input
        try:
            with lock:
                vectors = list(model.embed(inputs, batch_size=batch_size))
                tokens = model.token_count(inputs)
            if len(vectors) != len(inputs):
                raise ValueError("The model returned an unexpected number of embeddings")
            data = []
            for index, vector in enumerate(vectors):
                values = np.asarray(vector, dtype="<f4")
                if values.shape != (embedding_size,) or not np.isfinite(values).all():
                    raise ValueError("The model returned an invalid embedding")
                encoded = (
                    base64.b64encode(values.tobytes()).decode("ascii")
                    if body.encoding_format == "base64"
                    else values.tolist()
                )
                data.append({"object": "embedding", "index": index, "embedding": encoded})
            return {
                "object": "list",
                "data": data,
                "model": model.model_name,
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
            }
        except Exception:
            logger.exception("Embedding generation failed")
            return _error("Embedding generation failed.", status=500, error_type="server_error")

    return app
