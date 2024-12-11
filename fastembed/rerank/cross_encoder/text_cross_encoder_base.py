from abc import abstractclassmethod
from typing import Iterable, Optional, Any

from fastembed.common.model_management import ModelManagement


class TextCrossEncoderBase(ModelManagement):
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

    @classmethod
    def rerank(
        self,
        query: str,
        documents: Iterable[str],
        batch_size: int = 64,
        **kwargs,
    ) -> Iterable[float]:
        """Reranks a list of documents given a query.

        Args:
            query (str): The query to rerank the documents.
            documents (Iterable[str]): The list of texts to rerank.
            batch_size (int): The batch size to use for reranking.
            **kwargs: Additional keyword argument to pass to the rerank method.

        Yields:
            Iterable[float]: The scores of reranked the documents.
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    @classmethod
    def rerank_pairs(self, pairs: Iterable[tuple[str]], batch_size: int = 64,
        **kwargs: Any,) -> Iterable[float]:
        raise NotImplementedError("This method should be overridden by subclasses")

