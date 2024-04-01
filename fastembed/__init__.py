import importlib.metadata

from fastembed.text import TextEmbedding
from fastembed.sparse import SparseTextEmbedding, SparseEmbedding

__version__ = importlib.metadata.version("fastembed")
__all__ = ["TextEmbedding", "SparseTextEmbedding", "SparseEmbedding"]
