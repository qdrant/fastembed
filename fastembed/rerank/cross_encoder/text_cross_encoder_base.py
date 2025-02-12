from typing import Any, Iterable, Optional

from fastembed.common.model_description import BaseModelDescription
from fastembed.common.model_management import ModelManagement


class TextCrossEncoderBase(ModelManagement[BaseModelDescription]):
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

    def rerank(
        self,
        query: str,
        documents: Iterable[str],
        batch_size: int = 64,
        **kwargs: Any,
    ) -> Iterable[float]:
        """Rerank a list of documents given a query.

        Args:
            query (str): The query to rerank the documents.
            documents (Iterable[str]): The list of texts to rerank.
            batch_size (int): The batch size to use for reranking.
            **kwargs: Additional keyword argument to pass to the rerank method.

        Yields:
            Iterable[float]: The scores of the reranked the documents.
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def rerank_pairs(
        self,
        pairs: Iterable[tuple[str, str]],
        batch_size: int = 64,
        parallel: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterable[float]:
        """Rerank query-document pairs.
        Args:
            pairs (Iterable[tuple[str, str]]): Query-document pairs to rerank
            batch_size (int): The batch size to use for reranking.
            parallel: parallel:
                If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
                If 0, use all available cores.
                If None, don't use data-parallel processing, use default onnxruntime threading instead.
            **kwargs: Additional keyword argument to pass to the rerank method.
        Yields:
            Iterable[float]: Scores for each individual pair
        """
        raise NotImplementedError("This method should be overridden by subclasses")
