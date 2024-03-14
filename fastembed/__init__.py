import importlib.metadata

from fastembed.text.text_embedding import TextEmbedding

__version__ = importlib.metadata.version("fastembed")
__all__ = ["TextEmbedding"]
