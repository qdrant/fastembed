from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Union

import numpy as np

from fastembed.common.model_management import ModelManagement


@dataclass
class SparseEmbedding:
    values: np.ndarray
    indices: np.ndarray

    def as_object(self) -> Dict[str, np.ndarray]:
        return {
            "values": self.values,
            "indices": self.indices,
        }

    def as_dict(self) -> Dict[int, float]:
        return {i: v for i, v in zip(self.indices, self.values)}

    @classmethod
    def from_dict(cls, data: Dict[int, float]) -> "SparseEmbedding":
        if len(data) == 0:
            return cls(values=np.array([]), indices=np.array([]))
        indices, values = zip(*data.items())
        return cls(values=np.array(values), indices=np.array(indices))


class SparseTextEmbeddingBase(ModelManagement):
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
    ) -> Iterable[SparseEmbedding]:
        raise NotImplementedError()

    def passage_embed(
        self, texts: Iterable[str], **kwargs
    ) -> Iterable[SparseEmbedding]:
        """
        Embeds a list of text passages into a list of embeddings.

        Args:
            texts (Iterable[str]): The list of texts to embed.
            **kwargs: Additional keyword argument to pass to the embed method.

        Yields:
            Iterable[SparseEmbedding]: The sparse embeddings.
        """

        # This is model-specific, so that different models can have specialized implementations
        yield from self.embed(texts, **kwargs)

    def query_embed(
        self, query: Union[str, Iterable[str]], **kwargs
    ) -> Iterable[SparseEmbedding]:
        """
        Embeds queries

        Args:
            query (Union[str, Iterable[str]]): The query to embed, or an iterable e.g. list of queries.

        Returns:
            Iterable[SparseEmbedding]: The sparse embeddings.
        """

        # This is model-specific, so that different models can have specialized implementations
        if isinstance(query, str):
            yield from self.embed([query], **kwargs)
        if isinstance(query, Iterable):
            yield from self.embed(query, **kwargs)
