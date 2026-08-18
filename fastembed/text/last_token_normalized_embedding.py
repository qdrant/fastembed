from typing import Any, Iterable, Type

import onnxruntime as ort

from fastembed.common.types import NumpyArray
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.utils import last_token_pooling, normalize
from fastembed.text.onnx_embedding import OnnxTextEmbedding, OnnxTextEmbeddingWorker
from fastembed.common.model_description import DenseModelDescription, ModelSource

supported_last_token_normalized_models: list[DenseModelDescription] = [
    DenseModelDescription(
        model="Qwen/Qwen3-Embedding-0.6B",
        dim=1024,
        description=(
            "Text embeddings, Unimodal (text), multilingual, 32768 input tokens truncation, "
            "Prefixes for queries/documents: `Instruct: {task_description}\\nQuery:{query}` "
            "for queries, none for documents, 2025 year."
        ),
        license="apache-2.0",
        size_in_GB=2.38,
        sources=ModelSource(hf="Qdrant/Qwen3-Embedding-0.6B-onnx"),
        model_file="onnx/model.onnx",
        additional_files=["onnx/model.onnx.data"],
    ),
    DenseModelDescription(
        model="Qwen/Qwen3-Embedding-0.6B-Q",
        dim=1024,
        description=(
            "Text embeddings, Unimodal (text), multilingual, 32768 input tokens truncation, "
            "Prefixes for queries/documents: `Instruct: {task_description}\\nQuery:{query}` "
            "for queries, none for documents, int8 weights, requires onnxruntime>=1.23, "
            "2025 year."
        ),
        license="apache-2.0",
        size_in_GB=1.12,
        sources=ModelSource(hf="Qdrant/Qwen3-Embedding-0.6B-onnx"),
        model_file="onnx/model_quantized.onnx",
    ),
]


class LastTokenNormalizedEmbedding(OnnxTextEmbedding):
    """Decoder-based embedding models, which pool the last non-padding token and normalize it"""

    @classmethod
    def _get_worker_class(cls) -> Type[OnnxTextEmbeddingWorker]:
        return LastTokenNormalizedEmbeddingWorker

    def load_onnx_model(self) -> None:
        try:
            super().load_onnx_model()
        except Exception as e:
            # int8 weights are stored as 8-bit MatMulNBits, which onnxruntime only
            # implements since 1.23; older versions fail with "nbits_ == 4 was false"
            if "nbits" not in str(e).lower():
                raise
            raise RuntimeError(
                f"Could not load {self.model_name}: its int8 weights require "
                f"onnxruntime>=1.23, but onnxruntime {ort.__version__} is installed. "
                f"Either upgrade onnxruntime or use a non-quantized model."
            ) from e

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        """Lists the supported models.

        Returns:
            list[DenseModelDescription]: A list of DenseModelDescription objects containing the model information.
        """
        return supported_last_token_normalized_models

    def _post_process_onnx_output(
        self, output: OnnxOutputContext, **kwargs: Any
    ) -> Iterable[NumpyArray]:
        if output.attention_mask is None:
            raise ValueError("attention_mask must be provided for last token pooling")

        return normalize(last_token_pooling(output.model_output, output.attention_mask))


class LastTokenNormalizedEmbeddingWorker(OnnxTextEmbeddingWorker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs: Any,
    ) -> OnnxTextEmbedding:
        return LastTokenNormalizedEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
