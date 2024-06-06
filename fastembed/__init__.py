import importlib.metadata

from fastembed.image import ImageEmbedding
from fastembed.text import TextEmbedding
from fastembed.sparse import SparseTextEmbedding, SparseEmbedding
from fastembed.late_interaction import LateInteractionTextEmbedding

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
]
