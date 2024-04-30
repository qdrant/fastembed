from typing import Type, List, Dict, Any, Tuple, Iterable

import numpy as np

from fastembed.text.onnx_embedding import OnnxTextEmbedding, OnnxTextEmbeddingWorker
from fastembed.text.onnx_text_model import TextEmbeddingWorker

supported_clip_models = [
    {
        "model": "Qdrant/clip-ViT-B-32-text",
        "dim": 512,
        "description": "CLIP text encoder",
        "size_in_GB": 0.25,
        "sources": {
            "hf": "Qdrant/clip-ViT-B-32-text",
        },
        "model_file": "model.onnx",
    },
]


class CLIPOnnxEmbedding(OnnxTextEmbedding):
    @classmethod
    def _get_worker_class(cls) -> Type[TextEmbeddingWorker]:
        return CLIPEmbeddingWorker

    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        """Lists the supported models.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the model information.
        """
        return supported_clip_models

    @classmethod
    def _post_process_onnx_output(
        cls, output: Tuple[np.ndarray, np.ndarray]
    ) -> Iterable[np.ndarray]:
        embeddings, attention_mask = output
        return embeddings


class CLIPEmbeddingWorker(OnnxTextEmbeddingWorker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
    ) -> OnnxTextEmbedding:
        return CLIPOnnxEmbedding(model_name=model_name, cache_dir=cache_dir, threads=1)
