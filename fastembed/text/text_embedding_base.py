from typing import Iterable, Optional, Union

import numpy as np
from numpy.typing import NDArray

from fastembed.common.model_management import ModelManagement


class TextEmbeddingBase(ModelManagement):
    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.threads = threads
        self._local_files_only = kwargs.pop("local_files_only", False)

    def embed(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs,
    ) -> Iterable[NDArray[np.float32]]:
        raise NotImplementedError()

    def passage_embed(self, texts: Iterable[str], **kwargs) -> Iterable[NDArray[np.float32]]:
        """
        Embeds a list of text passages into a list of embeddings.

        Args:
            texts (Iterable[str]): The list of texts to embed.
            **kwargs: Additional keyword argument to pass to the embed method.

        Yields:
            Iterable[NDArray[np.float32]]: The embeddings.
        """

        # This is model-specific, so that different models can have specialized implementations
        yield from self.embed(texts, **kwargs)

    def query_embed(
        self, query: Union[str, Iterable[str]], **kwargs
    ) -> Iterable[NDArray[np.float32]]:
        """
        Embeds queries

        Args:
            query (Union[str, Iterable[str]]): The query to embed, or an iterable e.g. list of queries.

        Returns:
            Iterable[NDArray[np.float32]]: The embeddings.
        """

        # This is model-specific, so that different models can have specialized implementations
        if isinstance(query, str):
            yield from self.embed([query], **kwargs)
        if isinstance(query, Iterable):
            yield from self.embed(query, **kwargs)
