from typing import Any, Dict, Iterable, List, Type

import numpy as np

from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.text.onnx_embedding import OnnxTextEmbedding, OnnxTextEmbeddingWorker
from fastembed.text.onnx_text_model import TextEmbeddingWorker

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
]


class PooledEmbedding(OnnxTextEmbedding):
    @classmethod
    def _get_worker_class(cls) -> Type[TextEmbeddingWorker]:
        return PooledEmbeddingWorker

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
        return supported_pooled_models

    def _post_process_onnx_output(self, output: OnnxOutputContext) -> Iterable[np.ndarray]:
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
        **kwargs,
    ) -> OnnxTextEmbedding:
        return PooledEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
