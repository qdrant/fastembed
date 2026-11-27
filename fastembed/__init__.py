import importlib.metadata
from typing import Any

from fastembed import _telemetry as _telemetry  # first: opts out before onnxruntime loads
from fastembed.image import ImageEmbedding
from fastembed.late_interaction import LateInteractionTextEmbedding
from fastembed.late_interaction_multimodal import LateInteractionMultimodalEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from fastembed.sparse import SparseEmbedding, SparseTextEmbedding
from fastembed.text import TextEmbedding

try:
    version = importlib.metadata.version("fastembed")
except importlib.metadata.PackageNotFoundError as _:
    version = importlib.metadata.version("fastembed-gpu")


def list_supported_models() -> dict[str, list[dict[str, Any]]]:
    """Lists the supported models of every model class.

    Returns:
        dict[str, list[dict[str, Any]]]: Each class's supported models, keyed by class name.
    """
    model_classes = (
        TextEmbedding,
        SparseTextEmbedding,
        LateInteractionTextEmbedding,
        ImageEmbedding,
        LateInteractionMultimodalEmbedding,
        TextCrossEncoder,
    )
    return {
        model_class.__name__: model_class.list_supported_models() for model_class in model_classes
    }


__version__ = version
__all__ = [
    "TextEmbedding",
    "SparseTextEmbedding",
    "SparseEmbedding",
    "ImageEmbedding",
    "LateInteractionTextEmbedding",
    "LateInteractionMultimodalEmbedding",
    "list_supported_models",
]
