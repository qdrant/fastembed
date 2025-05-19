from typing import Iterable, Optional, Union, Any

from fastembed.common.model_description import DenseModelDescription
from fastembed.common.types import NumpyArray
from fastembed.common.model_management import ModelManagement


class TextEmbeddingBase(ModelManagement[DenseModelDescription]):
    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.threads = threads
        self._local_files_only = kwargs.pop("local_files_only", False)
        self._embedding_size: Optional[int] = None

    def embed(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        raise NotImplementedError()

    def passage_embed(self, texts: Iterable[str], **kwargs: Any) -> Iterable[NumpyArray]:
        """
        Embeds a list of text passages into a list of embeddings.

        Args:
            texts (Iterable[str]): The list of texts to embed.
            **kwargs: Additional keyword argument to pass to the embed method.

        Yields:
            Iterable[NumpyArray]: The embeddings.
        """

        # This is model-specific, so that different models can have specialized implementations
        yield from self.embed(texts, **kwargs)

    def query_embed(self, query: Union[str, Iterable[str]], **kwargs: Any) -> Iterable[NumpyArray]:
        """
        Embeds queries

        Args:
            query (Union[str, Iterable[str]]): The query to embed, or an iterable e.g. list of queries.

        Returns:
            Iterable[NumpyArray]: The embeddings.
        """

        # This is model-specific, so that different models can have specialized implementations
        if isinstance(query, str):
            yield from self.embed([query], **kwargs)
        else:
            yield from self.embed(query, **kwargs)

    @classmethod
    def get_embedding_size(cls, model_name: str) -> int:
        """Returns embedding size of the passed model."""
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def embedding_size(self) -> int:
        """Returns embedding size for the current model"""
        raise NotImplementedError("Subclasses must implement this method")
