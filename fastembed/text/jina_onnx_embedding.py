from typing import Type, List, Dict, Any, Tuple, Iterable

import numpy as np

from fastembed.common.models import normalize
from fastembed.common.onnx_model import EmbeddingWorker
from fastembed.text.onnx_embedding import OnnxTextEmbedding, OnnxTextEmbeddingWorker

supported_jina_models = [
    {
        "model": "jinaai/jina-embeddings-v2-base-en",
        "dim": 768,
        "description": "English embedding model supporting 8192 sequence length",
        "size_in_GB": 0.52,
        "sources": {"hf": "xenova/jina-embeddings-v2-base-en"},
        "model_file": "onnx/model.onnx",
    },
    {
        "model": "jinaai/jina-embeddings-v2-small-en",
        "dim": 512,
        "description": "English embedding model supporting 8192 sequence length",
        "size_in_GB": 0.12,
        "sources": {"hf": "xenova/jina-embeddings-v2-small-en"},
        "model_file": "onnx/model.onnx",
    },
]


class JinaOnnxEmbedding(OnnxTextEmbedding):
    @classmethod
    def _get_worker_class(cls) -> Type[EmbeddingWorker]:
        return JinaEmbeddingWorker

    @classmethod
    def mean_pooling(cls, model_output, attention_mask) -> np.ndarray:
        token_embeddings = model_output
        input_mask_expanded = (np.expand_dims(attention_mask, axis=-1)).astype(float)

        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        mask_sum = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)

        return sum_embeddings / mask_sum

    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        """Lists the supported models.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the model information.
        """
        return supported_jina_models

    @classmethod
    def _post_process_onnx_output(
        cls, output: Tuple[np.ndarray, np.ndarray]
    ) -> Iterable[np.ndarray]:
        embeddings, attn_mask = output
        return normalize(cls.mean_pooling(embeddings, attn_mask)).astype(np.float32)


class JinaEmbeddingWorker(OnnxTextEmbeddingWorker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
    ) -> OnnxTextEmbedding:
        return JinaOnnxEmbedding(model_name=model_name, cache_dir=cache_dir, threads=1)
