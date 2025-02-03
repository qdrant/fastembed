from typing import Any, Iterable, Type

import numpy as np

from fastembed.common.types import NdArray
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.text.onnx_embedding import OnnxTextEmbedding, OnnxTextEmbeddingWorker

supported_pooled_models = [
    {
        "model": "nomic-ai/nomic-embed-text-v1.5",
        "dim": 768,
        "description": "Text embeddings, Multimodal (text, image), English, 8192 input tokens truncation, Prefixes for queries/documents: necessary, 2024 year.",
        "license": "apache-2.0",
        "size_in_GB": 0.52,
        "sources": {
            "hf": "nomic-ai/nomic-embed-text-v1.5",
        },
        "model_file": "onnx/model.onnx",
    },
    {
        "model": "nomic-ai/nomic-embed-text-v1.5-Q",
        "dim": 768,
        "description": "Text embeddings, Multimodal (text, image), English, 8192 input tokens truncation, Prefixes for queries/documents: necessary, 2024 year.",
        "license": "apache-2.0",
        "size_in_GB": 0.13,
        "sources": {
            "hf": "nomic-ai/nomic-embed-text-v1.5",
        },
        "model_file": "onnx/model_quantized.onnx",
    },
    {
        "model": "nomic-ai/nomic-embed-text-v1",
        "dim": 768,
        "description": "Text embeddings, Multimodal (text, image), English, 8192 input tokens truncation, Prefixes for queries/documents: necessary, 2024 year.",
        "license": "apache-2.0",
        "size_in_GB": 0.52,
        "sources": {
            "hf": "nomic-ai/nomic-embed-text-v1",
        },
        "model_file": "onnx/model.onnx",
    },
    {
        "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "dim": 384,
        "description": "Text embeddings, Unimodal (text), Multilingual (~50 languages), 512 input tokens truncation, Prefixes for queries/documents: not necessary, 2019 year.",
        "license": "apache-2.0",
        "size_in_GB": 0.22,
        "sources": {
            "hf": "qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q",
        },
        "model_file": "model_optimized.onnx",
    },
    {
        "model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "dim": 768,
        "description": "Text embeddings, Unimodal (text), Multilingual (~50 languages), 384 input tokens truncation, Prefixes for queries/documents: not necessary, 2021 year.",
        "license": "apache-2.0",
        "size_in_GB": 1.00,
        "sources": {
            "hf": "xenova/paraphrase-multilingual-mpnet-base-v2",
        },
        "model_file": "onnx/model.onnx",
    },
    {
        "model": "intfloat/multilingual-e5-large",
        "dim": 1024,
        "description": "Text embeddings, Unimodal (text), Multilingual (~100 languages), 512 input tokens truncation, Prefixes for queries/documents: necessary, 2024 year.",
        "license": "mit",
        "size_in_GB": 2.24,
        "sources": {
            "url": "https://storage.googleapis.com/qdrant-fastembed/fast-multilingual-e5-large.tar.gz",
            "hf": "qdrant/multilingual-e5-large-onnx",
        },
        "model_file": "model.onnx",
        "additional_files": ["model.onnx_data"],
    },
]


class PooledEmbedding(OnnxTextEmbedding):
    @classmethod
    def _get_worker_class(cls) -> Type[OnnxTextEmbeddingWorker]:
        return PooledEmbeddingWorker

    @classmethod
    def mean_pooling(cls, model_output: NdArray, attention_mask: NdArray) -> NdArray:
        token_embeddings = model_output
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
        input_mask_expanded = np.tile(input_mask_expanded, (1, 1, token_embeddings.shape[-1]))
        input_mask_expanded = input_mask_expanded.astype(float)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.sum(input_mask_expanded, axis=1)
        pooled_embeddings = sum_embeddings / np.maximum(sum_mask, 1e-9)
        return pooled_embeddings

    @classmethod
    def list_supported_models(cls) -> list[dict[str, Any]]:
        """Lists the supported models.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing the model information.
        """
        return supported_pooled_models

    def _post_process_onnx_output(self, output: OnnxOutputContext) -> Iterable[NdArray]:
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
