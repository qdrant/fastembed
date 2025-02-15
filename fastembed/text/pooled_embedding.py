from typing import Any, Iterable, Type

import numpy as np

from fastembed.common.types import NumpyArray
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.text.onnx_embedding import OnnxTextEmbedding, OnnxTextEmbeddingWorker
from fastembed.common.model_description import DenseModelDescription, ModelSource

supported_pooled_models: list[DenseModelDescription] = [
    DenseModelDescription(
        model="nomic-ai/nomic-embed-text-v1.5",
        dim=768,
        description=(
            "Text embeddings, Multimodal (text, image), English, 8192 input tokens truncation, "
            "Prefixes for queries/documents: necessary, 2024 year."
        ),
        license="apache-2.0",
        size_in_GB=0.52,
        sources=ModelSource(hf="nomic-ai/nomic-embed-text-v1.5"),
        model_file="onnx/model.onnx",
    ),
    DenseModelDescription(
        model="nomic-ai/nomic-embed-text-v1.5-Q",
        dim=768,
        description=(
            "Text embeddings, Multimodal (text, image), English, 8192 input tokens truncation, "
            "Prefixes for queries/documents: necessary, 2024 year."
        ),
        license="apache-2.0",
        size_in_GB=0.13,
        sources=ModelSource(hf="nomic-ai/nomic-embed-text-v1.5"),
        model_file="onnx/model_quantized.onnx",
    ),
    DenseModelDescription(
        model="nomic-ai/nomic-embed-text-v1",
        dim=768,
        description=(
            "Text embeddings, Multimodal (text, image), English, 8192 input tokens truncation, "
            "Prefixes for queries/documents: necessary, 2024 year."
        ),
        license="apache-2.0",
        size_in_GB=0.52,
        sources=ModelSource(hf="nomic-ai/nomic-embed-text-v1"),
        model_file="onnx/model.onnx",
    ),
    DenseModelDescription(
        model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        dim=384,
        description=(
            "Text embeddings, Unimodal (text), Multilingual (~50 languages), 512 input tokens truncation, "
            "Prefixes for queries/documents: not necessary, 2019 year."
        ),
        license="apache-2.0",
        size_in_GB=0.22,
        sources=ModelSource(hf="qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"),
        model_file="model_optimized.onnx",
    ),
    DenseModelDescription(
        model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        dim=768,
        description=(
            "Text embeddings, Unimodal (text), Multilingual (~50 languages), 384 input tokens truncation, "
            "Prefixes for queries/documents: not necessary, 2021 year."
        ),
        license="apache-2.0",
        size_in_GB=1.00,
        sources=ModelSource(hf="xenova/paraphrase-multilingual-mpnet-base-v2"),
        model_file="onnx/model.onnx",
    ),
    DenseModelDescription(
        model="intfloat/multilingual-e5-large",
        dim=1024,
        description=(
            "Text embeddings, Unimodal (text), Multilingual (~100 languages), 512 input tokens truncation, "
            "Prefixes for queries/documents: necessary, 2024 year."
        ),
        license="mit",
        size_in_GB=2.24,
        sources=ModelSource(
            hf="qdrant/multilingual-e5-large-onnx",
            url="https://storage.googleapis.com/qdrant-fastembed/fast-multilingual-e5-large.tar.gz",
        ),
        model_file="model.onnx",
        additional_files=["model.onnx_data"],
    ),
]


class PooledEmbedding(OnnxTextEmbedding):
    @classmethod
    def _get_worker_class(cls) -> Type[OnnxTextEmbeddingWorker]:
        return PooledEmbeddingWorker

    @classmethod
    def mean_pooling(cls, model_output: NumpyArray, attention_mask: NumpyArray) -> NumpyArray:
        token_embeddings = model_output.astype(np.float32)
        attention_mask = attention_mask.astype(np.float32)
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
        input_mask_expanded = np.tile(input_mask_expanded, (1, 1, token_embeddings.shape[-1]))
        input_mask_expanded = input_mask_expanded.astype(np.float32)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.sum(input_mask_expanded, axis=1)
        pooled_embeddings = sum_embeddings / np.maximum(sum_mask, 1e-9)
        return pooled_embeddings

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        """Lists the supported models.

        Returns:
            list[DenseModelDescription]: A list of DenseModelDescription objects containing the model information.
        """
        return supported_pooled_models

    def _post_process_onnx_output(self, output: OnnxOutputContext) -> Iterable[NumpyArray]:
        if output.attention_mask is None:
            raise ValueError("attention_mask must be provided for document post-processing")

        embeddings = output.model_output
        attn_mask = output.attention_mask
        return self.mean_pooling(embeddings, attn_mask).astype(np.float32)


class PooledEmbeddingWorker(OnnxTextEmbeddingWorker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs: Any,
    ) -> OnnxTextEmbedding:
        return PooledEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
