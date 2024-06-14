from typing import Any, Dict, Iterable, List, Type

import numpy as np

from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.utils import normalize
from fastembed.text.onnx_embedding import OnnxTextEmbedding, OnnxTextEmbeddingWorker
from fastembed.text.onnx_text_model import TextEmbeddingWorker

supported_mini_lm_models = [
    {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "description": "Sentence Transformer model, MiniLM-L6-v2",
        "size_in_GB": 0.09,
        "sources": {
            "url": "https://storage.googleapis.com/qdrant-fastembed/sentence-transformers-all-MiniLM-L6-v2.tar.gz",
            "hf": "qdrant/all-MiniLM-L6-v2-onnx",
        },
        "model_file": "model.onnx",
    }
]


class MiniLMOnnxEmbedding(OnnxTextEmbedding):
    @classmethod
    def _get_worker_class(cls) -> Type[TextEmbeddingWorker]:
        return MiniLMEmbeddingWorker

    @classmethod
    def mean_pooling(cls, model_output: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        token_embeddings = model_output
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
        input_mask_expanded = np.tile(input_mask_expanded, (1, 1, token_embeddings.shape[-1]))
        input_mask_expanded = input_mask_expanded.astype(float)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.sum(input_mask_expanded, axis=1)
        pooled_embeddings = sum_embeddings / np.maximum(sum_mask, 1e-9)
        return pooled_embeddings

    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        """Lists the supported models.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the model information.
        """
        return supported_mini_lm_models

    def _post_process_onnx_output(self, output: OnnxOutputContext) -> Iterable[np.ndarray]:
        embeddings = output.model_output
        attn_mask = output.attention_mask
        return normalize(self.mean_pooling(embeddings, attn_mask)).astype(np.float32)


class MiniLMEmbeddingWorker(OnnxTextEmbeddingWorker):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs) -> OnnxTextEmbedding:
        return MiniLMOnnxEmbedding(model_name=model_name, cache_dir=cache_dir, threads=1, **kwargs)
