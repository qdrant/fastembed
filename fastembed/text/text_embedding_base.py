from typing import Optional, Union, Iterable, List, Dict, Any

import numpy as np

from fastembed.common.model_management import ModelManagement


class TextEmbeddingBase(ModelManagement):
    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        raise NotImplementedError()

    def __init__(self, model_name: str, cache_dir: Optional[str] = None, threads: Optional[int] = None, **kwargs):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.threads = threads

    def embed(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: int = None,
        **kwargs,
    ) -> Iterable[np.ndarray]:
        raise NotImplementedError()

    def passage_embed(self, texts: Iterable[str], **kwargs) -> Iterable[np.ndarray]:
        """
        Embeds a list of text passages into a list of embeddings.

        Args:
            texts (Iterable[str]): The list of texts to embed.
            **kwargs: Additional keyword argument to pass to the embed method.

        Yields:
            Iterable[np.ndarray]: The embeddings.
        """

        # This is model-specific, so that different models can have specialized implementations
        yield from self.embed(texts, **kwargs)

    def query_embed(self, query: str, **kwargs) -> np.ndarray:
        """
        Embeds a query

        Args:
            query (str): The query to search for.

        Returns:
            np.ndarray: The embeddings.
        """

        # This is model-specific, so that different models can have specialized implementations
        query_embedding = list(self.embed([query], **kwargs))[0]
        return query_embedding
