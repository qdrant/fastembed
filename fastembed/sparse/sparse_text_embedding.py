from typing import Any, Dict, Iterable, List, Optional, Sequence, Type, Union

from fastembed.common import OnnxProvider
from fastembed.sparse.bm25 import Bm25
from fastembed.sparse.bm42 import Bm42
from fastembed.sparse.sparse_embedding_base import (
    SparseEmbedding,
    SparseTextEmbeddingBase,
)
from fastembed.sparse.splade_pp import SpladePP
import warnings


class SparseTextEmbedding(SparseTextEmbeddingBase):
    EMBEDDINGS_REGISTRY: List[Type[SparseTextEmbeddingBase]] = [SpladePP, Bm42, Bm25]

    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        """
        Lists the supported models.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the model information.

            Example:
                ```
                [
                    {
                        "model": "prithvida/SPLADE_PP_en_v1",
                        "vocab_size": 30522,
                        "description": "Independent Implementation of SPLADE++ Model for English",
                        "license": "apache-2.0",
                        "size_in_GB": 0.532,
                        "sources": {
                            "hf": "qdrant/SPLADE_PP_en_v1",
                        },
                    }
                ]
                ```
        """
        result = []
        for embedding in cls.EMBEDDINGS_REGISTRY:
            result.extend(embedding.list_supported_models())
        return result

    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence[OnnxProvider]] = None,
        cuda: bool = False,
        device_ids: Optional[List[int]] = None,
        lazy_load: bool = False,
        **kwargs,
    ):
        super().__init__(model_name, cache_dir, threads, **kwargs)
        if model_name == "prithvida/Splade_PP_en_v1":
            warnings.warn(
                "The right spelling is prithivida/Splade_PP_en_v1. "
                "Support of this name will be removed soon, please fix the model_name",
                DeprecationWarning,
                stacklevel=2,
            )
            model_name = "prithivida/Splade_PP_en_v1"

        for EMBEDDING_MODEL_TYPE in self.EMBEDDINGS_REGISTRY:
            supported_models = EMBEDDING_MODEL_TYPE.list_supported_models()
            if any(model_name.lower() == model["model"].lower() for model in supported_models):
                self.model = EMBEDDING_MODEL_TYPE(
                    model_name,
                    cache_dir,
                    threads=threads,
                    providers=providers,
                    cuda=cuda,
                    device_ids=device_ids,
                    lazy_load=lazy_load,
                    **kwargs,
                )
                return

        raise ValueError(
            f"Model {model_name} is not supported in SparseTextEmbedding."
            "Please check the supported models using `SparseTextEmbedding.list_supported_models()`"
        )

    def embed(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs,
    ) -> Iterable[SparseEmbedding]:
        """
        Encode a list of documents into list of embeddings.
        We use mean pooling with attention so that the model can handle variable-length inputs.

        Args:
            documents: Iterator of documents or single document to embed
            batch_size: Batch size for encoding -- higher values will use more memory, but be faster
            parallel:
                If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
                If 0, use all available cores.
                If None, don't use data-parallel processing, use default onnxruntime threading instead.

        Returns:
            List of embeddings, one per document
        """
        yield from self.model.embed(documents, batch_size, parallel, **kwargs)

    def query_embed(self, query: Union[str, Iterable[str]], **kwargs) -> Iterable[SparseEmbedding]:
        """
        Embeds queries

        Args:
            query (Union[str, Iterable[str]]): The query to embed, or an iterable e.g. list of queries.

        Returns:
            Iterable[SparseEmbedding]: The sparse embeddings.
        """
        yield from self.model.query_embed(query, **kwargs)
