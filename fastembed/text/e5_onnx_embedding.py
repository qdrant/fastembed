from typing import Any, Dict, List, Type

import numpy as np

from fastembed.text.onnx_embedding import OnnxTextEmbedding, OnnxTextEmbeddingWorker
from fastembed.text.onnx_text_model import TextEmbeddingWorker

supported_multilingual_e5_models = [
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
]


class E5OnnxEmbedding(OnnxTextEmbedding):
    @classmethod
    def _get_worker_class(cls) -> Type["TextEmbeddingWorker"]:
        return E5OnnxEmbeddingWorker

    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        """Lists the supported models.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the model information.
        """
        return supported_multilingual_e5_models

    def _preprocess_onnx_input(
        self, onnx_input: Dict[str, np.ndarray], **kwargs
    ) -> Dict[str, np.ndarray]:
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
        **kwargs,
    ) -> E5OnnxEmbedding:
        return E5OnnxEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
