from typing import Any, Iterable, Type, Union, Optional

from fastembed.common import ImageInput
from fastembed.common.model_description import DenseModelDescription, ModelSource
from fastembed.common.onnx_model import OnnxOutputContext, T
from fastembed.common.types import NumpyArray
from fastembed.late_interaction_multimodal.late_interaction_multimodal_embedding_base import (
    LateInteractionMultimodalEmbeddingBase,
)
from fastembed.late_interaction_multimodal.onnx_multimodal_model import OnnxMultimodalModel, TextEmbeddingWorker, \
    ImageEmbeddingWorker

supported_colmodernvbert_models: list[DenseModelDescription] = [
    DenseModelDescription(
        model="Qdrant/colmodernvbert",
        dim=128,
        description="The late-interaction version of ModernVBERT, CPU friendly, English, 2025.",
        license="mit",
        size_in_GB=1.0,
        # TODO: change the url to hf repo link!
        sources=ModelSource(url="file:///home/kacper/Projects/Qdrant/colpali-model-migration-to-onnx/outputs/colmodernvbert"),
        additional_files=["model.onnx_data"],
        model_file="model.onnx",
    ),
]

class ColModernVBERT(LateInteractionMultimodalEmbeddingBase, OnnxMultimodalModel[NumpyArray]):
    """
    The ModernVBERT/colmodernvbert model implementation. This model uses
    bidirectional attention, which proves to work better for retrieval.

    See: https://huggingface.co/ModernVBERT/colmodernvbert
    """

    # TODO: reproduce ColPali methods only

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        """Lists the supported models.

        Returns:
            list[DenseModelDescription]: A list of DenseModelDescription objects containing the model information.
        """
        return supported_colmodernvbert_models

    @classmethod
    def _get_text_worker_class(cls) -> Type[TextEmbeddingWorker[NumpyArray]]:
        return ColModernVBERTTextEmbeddingWorker

    @classmethod
    def _get_image_worker_class(cls) -> Type[ImageEmbeddingWorker[NumpyArray]]:
        return ColModernVBERTmageEmbeddingWorker

class ColModernVBERTTextEmbeddingWorker(TextEmbeddingWorker[NumpyArray]):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs: Any) -> ColPali:
        return ColModernVBERT(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )


class ColModernVBERTmageEmbeddingWorker(ImageEmbeddingWorker[NumpyArray]):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs: Any) -> ColPali:
        return ColModernVBERT(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
