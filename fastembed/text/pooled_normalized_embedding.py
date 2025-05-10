from typing import Any, Iterable, Type


from fastembed.common.types import NumpyArray
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.utils import normalize
from fastembed.text.onnx_embedding import OnnxTextEmbedding, OnnxTextEmbeddingWorker
from fastembed.text.pooled_embedding import PooledEmbedding
from fastembed.common.model_description import DenseModelDescription, ModelSource

supported_pooled_normalized_models: list[DenseModelDescription] = [
    DenseModelDescription(
        model="sentence-transformers/all-MiniLM-L6-v2",
        dim=384,
        description=(
            "Text embeddings, Unimodal (text), English, 256 input tokens truncation, "
            "Prefixes for queries/documents: not necessary, 2021 year."
        ),
        license="apache-2.0",
        size_in_GB=0.09,
        sources=ModelSource(
            url="https://storage.googleapis.com/qdrant-fastembed/sentence-transformers-all-MiniLM-L6-v2.tar.gz",
            hf="qdrant/all-MiniLM-L6-v2-onnx",
            _deprecated_tar_struct=True,
        ),
        model_file="model.onnx",
    ),
    DenseModelDescription(
        model="jinaai/jina-embeddings-v2-base-en",
        dim=768,
        description=(
            "Text embeddings, Unimodal (text), English, 8192 input tokens truncation, "
            "Prefixes for queries/documents: not necessary, 2023 year."
        ),
        license="apache-2.0",
        size_in_GB=0.52,
        sources=ModelSource(hf="xenova/jina-embeddings-v2-base-en"),
        model_file="onnx/model.onnx",
    ),
    DenseModelDescription(
        model="jinaai/jina-embeddings-v2-small-en",
        dim=512,
        description=(
            "Text embeddings, Unimodal (text), English, 8192 input tokens truncation, "
            "Prefixes for queries/documents: not necessary, 2023 year."
        ),
        license="apache-2.0",
        size_in_GB=0.12,
        sources=ModelSource(hf="xenova/jina-embeddings-v2-small-en"),
        model_file="onnx/model.onnx",
    ),
    DenseModelDescription(
        model="jinaai/jina-embeddings-v2-base-de",
        dim=768,
        description=(
            "Text embeddings, Unimodal (text), Multilingual (German, English), 8192 input tokens truncation, "
            "Prefixes for queries/documents: not necessary, 2024 year."
        ),
        license="apache-2.0",
        size_in_GB=0.32,
        sources=ModelSource(hf="jinaai/jina-embeddings-v2-base-de"),
        model_file="onnx/model_fp16.onnx",
    ),
    DenseModelDescription(
        model="jinaai/jina-embeddings-v2-base-code",
        dim=768,
        description=(
            "Text embeddings, Unimodal (text), Multilingual (English, 30 programming languages), "
            "8192 input tokens truncation, Prefixes for queries/documents: not necessary, 2024 year."
        ),
        license="apache-2.0",
        size_in_GB=0.64,
        sources=ModelSource(hf="jinaai/jina-embeddings-v2-base-code"),
        model_file="onnx/model.onnx",
    ),
    DenseModelDescription(
        model="jinaai/jina-embeddings-v2-base-zh",
        dim=768,
        description=(
            "Text embeddings, Unimodal (text), supports mixed Chinese-English input text, "
            "8192 input tokens truncation, Prefixes for queries/documents: not necessary, 2024 year."
        ),
        license="apache-2.0",
        size_in_GB=0.64,
        sources=ModelSource(hf="jinaai/jina-embeddings-v2-base-zh"),
        model_file="onnx/model.onnx",
    ),
    DenseModelDescription(
        model="jinaai/jina-embeddings-v2-base-es",
        dim=768,
        description=(
            "Text embeddings, Unimodal (text), supports mixed Spanish-English input text, "
            "8192 input tokens truncation, Prefixes for queries/documents: not necessary, 2024 year."
        ),
        license="apache-2.0",
        size_in_GB=0.64,
        sources=ModelSource(hf="jinaai/jina-embeddings-v2-base-es"),
        model_file="onnx/model.onnx",
    ),
    DenseModelDescription(
        model="thenlper/gte-base",
        dim=768,
        description=(
            "General text embeddings, Unimodal (text), supports English only input text, "
            "512 input tokens truncation, Prefixes for queries/documents: not necessary, 2024 year."
        ),
        license="mit",
        size_in_GB=0.44,
        sources=ModelSource(hf="thenlper/gte-base"),
        model_file="onnx/model.onnx",
    ),
    DenseModelDescription(
        model="thenlper/gte-large",
        dim=1024,
        description=(
            "Text embeddings, Unimodal (text), English, 512 input tokens truncation, "
            "Prefixes for queries/documents: not necessary, 2023 year."
        ),
        license="mit",
        size_in_GB=1.20,
        sources=ModelSource(hf="qdrant/gte-large-onnx"),
        model_file="model.onnx",
    ),
]


class PooledNormalizedEmbedding(PooledEmbedding):
    @classmethod
    def _get_worker_class(cls) -> Type[OnnxTextEmbeddingWorker]:
        return PooledNormalizedEmbeddingWorker

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        """Lists the supported models.

        Returns:
            list[DenseModelDescription]: A list of DenseModelDescription objects containing the model information.
        """
        return supported_pooled_normalized_models

    def _post_process_onnx_output(
        self, output: OnnxOutputContext, **kwargs: Any
    ) -> Iterable[NumpyArray]:
        if output.attention_mask is None:
            raise ValueError("attention_mask must be provided for document post-processing")

        embeddings = output.model_output
        attn_mask = output.attention_mask
        return normalize(self.mean_pooling(embeddings, attn_mask))


class PooledNormalizedEmbeddingWorker(OnnxTextEmbeddingWorker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs: Any,
    ) -> OnnxTextEmbedding:
        return PooledNormalizedEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
