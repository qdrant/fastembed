from typing import Any, Type

from fastembed.common.types import NumpyArray
from fastembed.late_interaction.colbert import Colbert, ColbertEmbeddingWorker
from fastembed.common.model_description import DenseModelDescription, ModelSource

supported_jina_colbert_models: list[DenseModelDescription] = [
    DenseModelDescription(
        model="jinaai/jina-colbert-v2",
        dim=128,
        description="New model that expands capabilities of colbert-v1 with multilingual and context length of 8192, 2024 year",
        license="cc-by-nc-4.0",
        size_in_GB=2.24,
        sources=ModelSource(hf="jinaai/jina-colbert-v2"),
        model_file="onnx/model.onnx",
        additional_files=["onnx/model.onnx_data"],
    )
]


class JinaColbert(Colbert):
    QUERY_MARKER_TOKEN_ID = 250002
    DOCUMENT_MARKER_TOKEN_ID = 250003
    MIN_QUERY_LENGTH = 31  # it's 32, we add one additional special token in the beginning
    MASK_TOKEN = "<mask>"

    @classmethod
    def _get_worker_class(cls) -> Type[ColbertEmbeddingWorker]:
        return JinaColbertEmbeddingWorker

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        """Lists the supported models.

        Returns:
            list[DenseModelDescription]: A list of DenseModelDescription objects containing the model information.
        """
        return supported_jina_colbert_models

    def _preprocess_onnx_input(
        self, onnx_input: dict[str, NumpyArray], is_doc: bool = True, **kwargs: Any
    ) -> dict[str, NumpyArray]:
        onnx_input = super()._preprocess_onnx_input(onnx_input, is_doc)

        # the attention mask for jina-colbert-v2 is always 1 in queries
        if not is_doc:
            onnx_input["attention_mask"][:] = 1
        return onnx_input


class JinaColbertEmbeddingWorker(ColbertEmbeddingWorker):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs: Any) -> JinaColbert:
        return JinaColbert(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
