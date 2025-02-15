from dataclasses import dataclass
from typing import Iterable, Optional, Union, Any

import numpy as np
from numpy.typing import NDArray

from fastembed.common.model_description import SparseModelDescription
from fastembed.common.types import NumpyArray
from fastembed.common.model_management import ModelManagement


@dataclass
class SparseEmbedding:
    values: NumpyArray
    indices: Union[NDArray[np.int64], NDArray[np.int32]]

    def as_object(self) -> dict[str, NumpyArray]:
        return {
            "values": self.values,
            "indices": self.indices,
        }

    def as_dict(self) -> dict[int, float]:
        return {int(i): float(v) for i, v in zip(self.indices, self.values)}  # type: ignore

    @classmethod
    def from_dict(cls, data: dict[int, float]) -> "SparseEmbedding":
        if len(data) == 0:
            return cls(values=np.array([]), indices=np.array([]))
        indices, values = zip(*data.items())
        return cls(values=np.array(values), indices=np.array(indices))


class SparseTextEmbeddingBase(ModelManagement[SparseModelDescription]):
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

    def embed(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterable[SparseEmbedding]:
        raise NotImplementedError()

    def passage_embed(self, texts: Iterable[str], **kwargs: Any) -> Iterable[SparseEmbedding]:
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
        self, query: Union[str, Iterable[str]], **kwargs: Any
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
        else:
            yield from self.embed(query, **kwargs)
