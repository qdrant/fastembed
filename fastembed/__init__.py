import importlib.metadata
import os

# onnxruntime records usage telemetry (process, model load, session and device events, with a
# persistent device id) and queues it for upload to Microsoft. Importing onnxruntime already
# queues events, so the opt-out has to be in the environment before its first import.
# ORT_DISABLE_TELEMETRY=0 set by the user keeps onnxruntime's default.
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
