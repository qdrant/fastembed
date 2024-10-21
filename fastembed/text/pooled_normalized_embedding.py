from typing import Any, Dict, Iterable, List, Type

import numpy as np

from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.utils import normalize
from fastembed.text.onnx_embedding import OnnxTextEmbedding, OnnxTextEmbeddingWorker
from fastembed.text.onnx_text_model import TextEmbeddingWorker
from fastembed.text.pooled_embedding import PooledEmbedding

supported_pooled_normalized_models = [
    {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "description": "Text embeddings, Unimodal (text), English, 256 input tokens truncation, Prefixes for queries/documents: not necessary, 2021 year.",
        "license": "apache-2.0",
        "size_in_GB": 0.09,
        "sources": {
            "url": "https://storage.googleapis.com/qdrant-fastembed/sentence-transformers-all-MiniLM-L6-v2.tar.gz",
            "hf": "qdrant/all-MiniLM-L6-v2-onnx",
        },
        "model_file": "model.onnx",
    },
    {
        "model": "jinaai/jina-embeddings-v2-base-en",
        "dim": 768,
        "description": "Text embeddings, Unimodal (text), English, 8192 input tokens truncation, Prefixes for queries/documents: not necessary, 2023 year.",
        "license": "apache-2.0",
        "size_in_GB": 0.52,
        "sources": {"hf": "xenova/jina-embeddings-v2-base-en"},
        "model_file": "onnx/model.onnx",
    },
    {
        "model": "jinaai/jina-embeddings-v2-small-en",
        "dim": 512,
        "description": "Text embeddings, Unimodal (text), English, 8192 input tokens truncation, Prefixes for queries/documents: not necessary, 2023 year.",
        "license": "apache-2.0",
        "size_in_GB": 0.12,
        "sources": {"hf": "xenova/jina-embeddings-v2-small-en"},
        "model_file": "onnx/model.onnx",
    },
    {
        "model": "jinaai/jina-embeddings-v2-base-de",
        "dim": 768,
        "description": "Text embeddings, Unimodal (text), Multilingual (German, English), 8192 input tokens truncation, Prefixes for queries/documents: not necessary, 2024 year.",
        "license": "apache-2.0",
        "size_in_GB": 0.32,
        "sources": {"hf": "jinaai/jina-embeddings-v2-base-de"},
        "model_file": "onnx/model_fp16.onnx",
    },
    {
        "model": "jinaai/jina-embeddings-v2-base-code",
        "dim": 768,
        "description": "Text embeddings, Unimodal (text), Multilingual (English, 30 programming languages), 8192 input tokens truncation, Prefixes for queries/documents: not necessary, 2024 year.",
        "license": "apache-2.0",
        "size_in_GB": 0.64,
        "sources": {"hf": "jinaai/jina-embeddings-v2-base-code"},
        "model_file": "onnx/model.onnx",
    },
]


class PooledNormalizedEmbedding(PooledEmbedding):
    @classmethod
    def _get_worker_class(cls) -> Type[TextEmbeddingWorker]:
        return PooledNormalizedEmbeddingWorker

    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        """Lists the supported models.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the model information.
        """
        return supported_pooled_normalized_models

    def _post_process_onnx_output(self, output: OnnxOutputContext) -> Iterable[np.ndarray]:
        if output.attention_mask is None:
            raise ValueError("attention_mask must be provided for document post-processing")

        embeddings = output.model_output
        attn_mask = output.attention_mask
        return normalize(self.mean_pooling(embeddings, attn_mask)).astype(np.float32)


class PooledNormalizedEmbeddingWorker(OnnxTextEmbeddingWorker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs,
    ) -> OnnxTextEmbedding:
        return PooledNormalizedEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
