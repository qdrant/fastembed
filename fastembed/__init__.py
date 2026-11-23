import importlib.metadata
import os

# onnxruntime queues telemetry for Microsoft from its first import on; opt out before it.
# A user's own ORT_DISABLE_TELEMETRY (e.g. 0) is kept.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

from fastembed.image import ImageEmbedding
from fastembed.late_interaction import LateInteractionTextEmbedding
from fastembed.late_interaction_multimodal import LateInteractionMultimodalEmbedding
from fastembed.sparse import SparseEmbedding, SparseTextEmbedding
from fastembed.text import TextEmbedding

try:
    version = importlib.metadata.version("fastembed")
except importlib.metadata.PackageNotFoundError as _:
    version = importlib.metadata.version("fastembed-gpu")

__version__ = version
__all__ = [
    "TextEmbedding",
    "SparseTextEmbedding",
    "SparseEmbedding",
    "ImageEmbedding",
    "LateInteractionTextEmbedding",
    "LateInteractionMultimodalEmbedding",
]
