from typing import Any, Dict, Iterable, List, Optional, Sequence, Type, Union

import numpy as np

from fastembed.common import OnnxProvider
from fastembed.late_interaction.colbert import Colbert
from fastembed.late_interaction.jina_colbert import JinaColbert
from fastembed.late_interaction.late_interaction_embedding_base import (
    LateInteractionTextEmbeddingBase,
)


class LateInteractionTextEmbedding(LateInteractionTextEmbeddingBase):
    EMBEDDINGS_REGISTRY: List[Type[LateInteractionTextEmbeddingBase]] = [Colbert, JinaColbert]

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
                        "model": "colbert-ir/colbertv2.0",
                        "dim": 128,
                        "description": "Late interaction model",
                        "license": "mit",
                        "size_in_GB": 0.44,
                        "sources": {
                            "hf": "colbert-ir/colbertv2.0",
                        },
                        "model_file": "model.onnx",
                    },
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
            f"Model {model_name} is not supported in LateInteractionTextEmbedding."
            "Please check the supported models using `LateInteractionTextEmbedding.list_supported_models()`"
        )

    def embed(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs,
    ) -> Iterable[np.ndarray]:
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

    def query_embed(self, query: Union[str, Iterable[str]], **kwargs) -> Iterable[np.ndarray]:
        """
        Embeds queries

        Args:
            query (Union[str, Iterable[str]]): The query to embed, or an iterable e.g. list of queries.

        Returns:
            Iterable[np.ndarray]: The embeddings.
        """

        # This is model-specific, so that different models can have specialized implementations
        yield from self.model.query_embed(query, **kwargs)
