from typing import Any, Dict, List, Tuple, Type

import numpy as np

from fastembed.text.onnx_embedding import (
    EmbeddingWorker,
    OnnxTextEmbedding,
    OnnxTextEmbeddingWorker,
)

supported_splade_models = [
    {
        "model": "prithvida/SPLADE_PP_en_v1",
        "dim": 30552,  # vocab size
        "description": "Independent Implementation of SPLADE++ Model for English",
        "size_in_GB": 0.532,
        "sources": {
            "hf": "nirantk/SPLADE_PP_en_v1",
        },
    },
]


class SPLADE(OnnxTextEmbedding):
    @classmethod
    def _get_worker_class(cls) -> Type["EmbeddingWorker"]:
        return SPLADEOnnxEmbeddingWorker

    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        """Lists the supported models.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the model information.
        """
        return supported_splade_models

    def _preprocess_onnx_input(self, onnx_input: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Preprocess the onnx input.
        """
        return onnx_input

    @classmethod
    def _post_process_onnx_output(cls, output: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        logits, attention_mask = output
        relu_log = np.log(1 + np.maximum(logits, 0))

        weighted_log = relu_log * np.expand_dims(attention_mask, axis=-1)

        max_val = np.max(weighted_log, axis=1)
        binary_vector = np.squeeze(max_val)
        return binary_vector


class SPLADEOnnxEmbeddingWorker(OnnxTextEmbeddingWorker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
    ) -> SPLADE:
        return SPLADE(model_name=model_name, cache_dir=cache_dir, threads=1)
