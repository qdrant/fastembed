from typing import Any, Dict, Iterable, List

import numpy as np

from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.text.onnx_embedding import OnnxTextEmbedding

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
        "additional_files": [
            "model.onnx_data",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "config.json",
            "preprocessor_config.json",
        ],
        "model_file": "model.onnx",
    }
]


class ColpaliTextModel(OnnxTextEmbedding):
    query_prefix = "Query: "
    bos_token = "<s>"
    pad_token = "<pad>"

    def _preprocess_onnx_input(
        self, onnx_input: Dict[str, np.ndarray], **kwargs
    ) -> Dict[str, np.ndarray]:
        empty_image_placeholder = np.zeros((3, 448, 448), dtype=np.float32)
        onnx_input["pixel_values"] = np.array(
            [empty_image_placeholder for _ in onnx_input["input_ids"]]
        )
        onnx_input["attention_mask"] = np.array([[1] for _ in onnx_input["input_ids"]])
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

    def _preprocess_queries(self, documents: List[str]):
        texts_query: List[str] = []

        for query in documents:
            query = self.bos_token + self.query_prefix + query + self.pad_token * 10
            query += "\n"

            texts_query.append(query)
        return texts_query

    def onnx_embed(
        self,
        documents: List[str],
        **kwargs,
    ) -> OnnxOutputContext:
        documents = self._preprocess_queries(documents)
        self.tokenizer.enable_truncation(max_length=10000)
        encoded = self.tokenize(documents, **kwargs)
        input_ids = np.array([[2, 9413] + e.ids[2:] for e in encoded])

        attention_mask = np.array([e.attention_mask for e in encoded])
        onnx_input = {"input_ids": np.array(input_ids, dtype=np.int64)}
        onnx_input = self._preprocess_onnx_input(onnx_input, **kwargs)
        onnx_input["attention_mask"] = attention_mask
        model_output = self.model.run(self.ONNX_OUTPUT_NAMES, onnx_input)
        return OnnxOutputContext(
            model_output=model_output[0],
            attention_mask=onnx_input.get("attention_mask", attention_mask),
            input_ids=onnx_input.get("input_ids", input_ids),
        )
