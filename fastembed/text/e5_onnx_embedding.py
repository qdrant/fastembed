from typing import Type, List, Dict, Any

import numpy as np

from fastembed.common.onnx_model import EmbeddingWorker
from fastembed.text.onnx_embedding import OnnxTextEmbedding, OnnxTextEmbeddingWorker

supported_multilingual_e5_models = [
    {
        "model": "intfloat/multilingual-e5-large",
        "dim": 1024,
        "description": "Multilingual model, e5-large. Recommend using this model for non-English languages",
        "size_in_GB": 2.24,
        "sources": {
            "url": "https://storage.googleapis.com/qdrant-fastembed/fast-multilingual-e5-large.tar.gz",
            "hf": "qdrant/multilingual-e5-large-onnx",
        },
    },
    {
        "model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "dim": 768,
        "description": "Sentence-transformers model for tasks like clustering or semantic search",
        "size_in_GB": 1.00,
        "sources": {
            "hf": "xenova/paraphrase-multilingual-mpnet-base-v2",
        },
    },
    {
        "model": "intfloat/multilingual-e5-large-instruct",
        "dim": 1024,
        "description": "multilingual model, e5-large-instruct",
        "size_in_GB": 1.03,
        "sources": {
            "url": "https://huggingface.co/yashvardhan7/multilingual-e5-large-instruct-onnx/blob/main/fast-multilingual-e5-large-instruct%20.tar.gz",
            "hf": "multilingual-e5-large-instruct-onnx",
        },
    },
    {
        "model": "intfloat/multilingual-e5-small",
        "dim": 384,
        "description": "multilingual model, e5-small",
        "size_in_GB": 0.492,
        "sources": {
            "url": "https://huggingface.co/intfloat/multilingual-e5-small/tree/main/onnx",
            "hf": "multilingual-e5-small-onnx",
        },
    },
    {
        "model": "dangvantuan/sentence-camembert-base",
        "dim": 768,
        "description": "French embedding model",
        "size_in_GB": 0.445,
        "sources": {
            "url": "https://huggingface.co/yashvardhan7/fast-sentence-camembert-base/blob/main/fast-sentence-camembert-base.tar.gz",
            "hf": "sentence-camembert-base-onnx",
        },
    },
]


class E5OnnxEmbedding(OnnxTextEmbedding):
    @classmethod
    def _get_worker_class(cls) -> Type["EmbeddingWorker"]:
        return E5OnnxEmbeddingWorker

    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        """Lists the supported models.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the model information.
        """
        return supported_multilingual_e5_models

    def _preprocess_onnx_input(self, onnx_input: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Preprocess the onnx input.
        """
        onnx_input.pop("token_type_ids", None)
        return onnx_input


class E5OnnxEmbeddingWorker(OnnxTextEmbeddingWorker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
    ) -> E5OnnxEmbedding:
        return E5OnnxEmbedding(model_name=model_name, cache_dir=cache_dir, threads=1)
