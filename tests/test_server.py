import base64
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Any

import numpy as np
import pytest

from fastembed.common.types import NumpyArray
from fastembed.text.text_embedding_base import TextEmbeddingBase

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from fastembed.server import create_app  # noqa: E402


class ExampleEmbedding(TextEmbeddingBase):
    def __init__(self) -> None:
        super().__init__(model_name="test-model")
        self.calls: list[tuple[list[str], int]] = []

    @property
    def embedding_size(self) -> int:
        return 2

    def embed(
        self, documents: Iterable[str], batch_size: int = 256, **kwargs: Any
    ) -> Iterable[NumpyArray]:
        texts = list(documents)
        self.calls.append((texts, batch_size))
        for text in texts:
            yield np.array([len(text), len(text.split())], dtype=np.float32)

    def token_count(self, texts: str | Iterable[str], **kwargs: Any) -> int:
        return sum(len(text.split()) for text in ([texts] if isinstance(texts, str) else texts))


@pytest.fixture
def model() -> ExampleEmbedding:
    return ExampleEmbedding()


@pytest.mark.parametrize("inputs", ["one two", ["one two", "three"]])
@pytest.mark.parametrize("encoding", ["float", "base64"])
def test_embeddings(model: ExampleEmbedding, inputs: str | list[str], encoding: str) -> None:
    with TestClient(create_app(model, batch_size=8)) as client:
        response = client.post(
            "/v1/embeddings",
            json={
                "input": inputs,
                "model": "test-model",
                "encoding_format": encoding,
                "dimensions": 2,
            },
        )
    assert response.status_code == 200
    result = response.json()
    texts = [inputs] if isinstance(inputs, str) else inputs
    assert result["object"] == "list"
    assert result["model"] == "test-model"
    tokens = 2 if isinstance(inputs, str) else 3
    assert result["usage"] == {"prompt_tokens": tokens, "total_tokens": tokens}
    assert len(result["data"]) == len(texts)
    for index, (item, text) in enumerate(zip(result["data"], texts)):
        assert item["object"] == "embedding"
        assert item["index"] == index
        values = item["embedding"]
        if encoding == "base64":
            values = np.frombuffer(base64.b64decode(values), dtype="<f4").tolist()
        assert values == [len(text), len(text.split())]
    assert model.calls == [(texts, 8)]


def test_default_model_and_encoding(model: ExampleEmbedding) -> None:
    with TestClient(create_app(model)) as client:
        response = client.post("/v1/embeddings", json={"input": "one", "user": "local-user"})
        listed = client.get("/v1/models").json()
    assert response.status_code == 200
    assert response.json()["data"][0]["embedding"] == [3, 1]
    assert listed["object"] == "list"
    assert listed["data"][0]["id"] == "test-model"


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"input": ""},
        {"input": []},
        {"input": [""]},
        {"input": None},
        {"input": 12},
        {"input": [1, 2]},
        {"input": [[1, 2]]},
        {"input": ["private text", 12]},
        {"input": "private text", "encoding_format": "invalid"},
        {"input": "private text", "dimensions": True},
        {"input": "private text", "dimensions": 0},
        {"input": "private text", "dimensions": "2"},
        {"input": "private text", "extra": "ignored?"},
        {"input": ["one"] * 2049},
    ],
)
def test_invalid_requests_do_not_run_inference(model: ExampleEmbedding, body: dict) -> None:
    with TestClient(create_app(model)) as client:
        response = client.post("/v1/embeddings", json=body)
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "private text" not in response.text
    assert not model.calls


def test_malformed_json(model: ExampleEmbedding) -> None:
    with TestClient(create_app(model)) as client:
        response = client.post(
            "/v1/embeddings", content='{"input":', headers={"content-type": "application/json"}
        )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert response.json()["error"]["param"] is None


@pytest.mark.parametrize(
    ("extra", "status", "param"),
    [({"model": "not-loaded"}, 404, "model"), ({"dimensions": 1}, 400, "dimensions")],
)
def test_unsupported_model_or_dimensions(
    model: ExampleEmbedding, extra: dict, status: int, param: str
) -> None:
    with TestClient(create_app(model)) as client:
        response = client.post("/v1/embeddings", json={"input": "one", **extra})
    assert response.status_code == status
    assert response.json()["error"]["param"] == param
    assert not model.calls


@pytest.mark.parametrize("failure", ["exception", "missing", "nan", "shape"])
def test_inference_failure_is_not_exposed(
    model: ExampleEmbedding, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    def broken_embed(*args: Any, **kwargs: Any) -> Iterable[NumpyArray]:
        if failure == "exception":
            raise RuntimeError("private model path and input")
        if failure == "missing":
            return []
        if failure == "nan":
            return [np.array([np.nan, 1])]
        return [np.array([[1, 2]])]

    monkeypatch.setattr(model, "embed", broken_embed)
    with TestClient(create_app(model)) as client:
        response = client.post("/v1/embeddings", json={"input": "one"})
    assert response.status_code == 500
    assert response.json()["error"]["type"] == "server_error"
    assert "private" not in response.text


def test_health_during_inference(model: ExampleEmbedding, monkeypatch: pytest.MonkeyPatch) -> None:
    started, release = Event(), Event()
    original_embed = model.embed

    def blocking_embed(*args: Any, **kwargs: Any) -> Iterable[NumpyArray]:
        started.set()
        if not release.wait(timeout=10):
            raise TimeoutError("Inference was not released")
        yield from original_embed(*args, **kwargs)

    monkeypatch.setattr(model, "embed", blocking_embed)

    async def limit_workers() -> None:
        import anyio.to_thread

        # One blocked inference exhausts this test server's entire worker pool.
        anyio.to_thread.current_default_thread_limiter().total_tokens = 1

    with TestClient(create_app(model)) as client, ThreadPoolExecutor(max_workers=2) as pool:
        assert client.portal is not None
        client.portal.call(limit_workers)
        pending = pool.submit(client.post, "/v1/embeddings", json={"input": "one"})
        try:
            assert started.wait(timeout=5)
            health = pool.submit(client.get, "/health").result(timeout=5)
            assert health.status_code == 200
            assert health.json() == {"status": "ok"}
            listed = pool.submit(client.get, "/v1/models").result(timeout=5)
            assert listed.json()["data"][0]["id"] == "test-model"
        finally:
            release.set()
        assert pending.result(timeout=5).status_code == 200


def test_invalid_batch_size(model: ExampleEmbedding) -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        create_app(model, batch_size=0)
