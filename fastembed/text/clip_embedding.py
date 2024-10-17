from typing import Any, Dict, Iterable, List, Type

import numpy as np

from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.text.onnx_embedding import OnnxTextEmbedding, OnnxTextEmbeddingWorker
from fastembed.text.onnx_text_model import TextEmbeddingWorker

supported_clip_models = [
    {
        "model": "Qdrant/clip-ViT-B-32-text",
        "dim": 512,
        "description": "Text embeddings, Multimodal (text&image), English, 77 input tokens truncation, Prefixes for queries/documents: not necessary, 2021 year",
        "license": "mit",
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

    def _post_process_onnx_output(self, output: OnnxOutputContext) -> Iterable[np.ndarray]:
        return output.model_output


class CLIPEmbeddingWorker(OnnxTextEmbeddingWorker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs,
    ) -> OnnxTextEmbedding:
        return CLIPOnnxEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
