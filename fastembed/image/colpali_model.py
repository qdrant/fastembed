import contextlib
from typing import Any, Dict, Iterable, List

import numpy as np
from PIL import Image

from fastembed.common import ImageInput
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.image.onnx_embedding import OnnxImageEmbedding

supported_onnx_models = [
    {
        "model": "akshayballal/colpali-v1.2-merged",
        "dim": 128,
        "description": "",
        "license": "mit",
        "size_in_GB": 6.08,
        "sources": {
            "hf": "akshayballal/colpali-v1.2-merged-onnx",
        },
        "additional_files": ["model.onnx_data"],
        "model_file": "model.onnx",
    }
]


class ColpaliImageModel(OnnxImageEmbedding):
    def _preprocess_onnx_input(
        self, onnx_input: Dict[str, np.ndarray], **kwargs
    ) -> Dict[str, np.ndarray]:
        empty_text_placeholder = np.array([257152] * 1024 + [2, 50721, 573, 2416, 235265, 108])
        even_attention_mask = np.array([1] * 1030)
        onnx_input["input_ids"] = np.array(
            [empty_text_placeholder for _ in onnx_input["input_ids"]]
        )
        onnx_input["attention_mask"] = np.array(
            [even_attention_mask for _ in onnx_input["input_ids"]]
        )
        return onnx_input

    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        """
        Lists the supported models.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the model information.
        """
        return supported_onnx_models

    def _post_process_onnx_output(self, output: OnnxOutputContext) -> Iterable[np.ndarray]:
        return output.model_output.astype(np.float32)

    def onnx_embed(self, images: List[ImageInput], **kwargs) -> OnnxOutputContext:
        with contextlib.ExitStack():
            image_files = [
                Image.open(image) if not isinstance(image, Image.Image) else image
                for image in images
            ]
            encoded = self.processor(image_files)
        onnx_input = self._build_onnx_input(encoded)
        onnx_input = self._preprocess_onnx_input(onnx_input)

        model_output = self.model.run(None, onnx_input)
        embeddings = model_output[0].reshape(len(images), -1, supported_onnx_models[0]["dim"])
        return OnnxOutputContext(model_output=embeddings)
