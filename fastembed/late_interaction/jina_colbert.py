from typing import Any, Dict, List, Type

import numpy as np

from fastembed.late_interaction.colbert import Colbert
from fastembed.text.onnx_text_model import TextEmbeddingWorker


supported_jina_colbert_models = [
    {
        "model": "jinaai/jina-colbert-v2",
        "dim": 128,
        "description": "New model that expands capabilities of colbert-v1 with multilingual and context length of 8192, 2024 year",
        "license": "cc-by-nc-4.0",
        "size_in_GB": 2.24,
        "sources": {
            "hf": "jinaai/jina-colbert-v2",
        },
        "model_file": "onnx/model.onnx",
        "additional_files": ["onnx/model.onnx_data"],
    },
]


class JinaColbert(Colbert):
    QUERY_MARKER_TOKEN_ID = 250002
    DOCUMENT_MARKER_TOKEN_ID = 250003
    MIN_QUERY_LENGTH = 31  # it's 32, we add one additional special token in the beginning
    MASK_TOKEN = "<mask>"

    @classmethod
    def _get_worker_class(cls) -> Type[TextEmbeddingWorker]:
        return JinaColbertEmbeddingWorker

    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        """Lists the supported models.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the model information.
        """
        return supported_jina_colbert_models

    def _preprocess_onnx_input(
        self, onnx_input: Dict[str, np.ndarray], is_doc: bool = True, **kwargs: Any
    ) -> Dict[str, np.ndarray]:
        onnx_input = super()._preprocess_onnx_input(onnx_input, is_doc)

        # the attention mask for jina-colbert-v2 is always 1 in queries
        if not is_doc:
            onnx_input["attention_mask"][:] = 1
        return onnx_input


class JinaColbertEmbeddingWorker(TextEmbeddingWorker):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs) -> JinaColbert:
        return JinaColbert(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
