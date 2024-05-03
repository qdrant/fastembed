import importlib.metadata

from fastembed.text import TextEmbedding
from fastembed.sparse import SparseTextEmbedding, SparseEmbedding

try:
    version = importlib.metadata.version("fastembed")
except importlib.metadata.PackageNotFoundError as _:
    version = importlib.metadata.version("fastembed-gpu")

__version__ = version
__all__ = ["TextEmbedding", "SparseTextEmbedding", "SparseEmbedding"]
