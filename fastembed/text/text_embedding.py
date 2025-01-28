import warnings
from typing import Any, Iterable, Optional, Sequence, Type, Union

import numpy as np
from numpy.typing import NDArray

from fastembed.common import OnnxProvider
from fastembed.text.clip_embedding import CLIPOnnxEmbedding
from fastembed.text.e5_onnx_embedding import E5OnnxEmbedding
from fastembed.text.pooled_normalized_embedding import PooledNormalizedEmbedding
from fastembed.text.pooled_embedding import PooledEmbedding
from fastembed.text.multitask_embedding import JinaEmbeddingV3
from fastembed.text.onnx_embedding import OnnxTextEmbedding
from fastembed.text.text_embedding_base import TextEmbeddingBase


class TextEmbedding(TextEmbeddingBase):
    EMBEDDINGS_REGISTRY: list[Type[TextEmbeddingBase]] = [
        OnnxTextEmbedding,
        E5OnnxEmbedding,
        CLIPOnnxEmbedding,
        PooledNormalizedEmbedding,
        PooledEmbedding,
        JinaEmbeddingV3,
    ]

    @classmethod
    def list_supported_models(cls) -> list[dict[str, Any]]:
        """
        Lists the supported models.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing the model information.

            Example:
                ```
                [
                    {
                        "model": "intfloat/multilingual-e5-large",
                        "dim": 1024,
                        "description": "Multilingual model, e5-large. Recommend using this model for non-English languages",
                        "license": "mit",
                        "size_in_GB": 2.24,
                        "sources": {
                            "gcp": "https://storage.googleapis.com/qdrant-fastembed/fast-multilingual-e5-large.tar.gz",
                            "hf": "qdrant/multilingual-e5-large-onnx",
                        }
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
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence[OnnxProvider]] = None,
        cuda: bool = False,
        device_ids: Optional[list[int]] = None,
        lazy_load: bool = False,
        **kwargs,
    ):
        super().__init__(model_name, cache_dir, threads, **kwargs)
        if model_name == "nomic-ai/nomic-embed-text-v1.5-Q":
            warnings.warn(
                "The model 'nomic-ai/nomic-embed-text-v1.5-Q' has been updated on HuggingFace. "
                "Please review the latest documentation and release notes to ensure compatibility with your workflow. ",
                UserWarning,
                stacklevel=2,
            )
        if model_name == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2":
            warnings.warn(
                "The model 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' has been updated to "
                "include a mean pooling layer. Please ensure your usage aligns with the new functionality. "
                "Support for the previous version without mean pooling will be removed as of version 0.5.2.",
                UserWarning,
                stacklevel=2,
            )
        for EMBEDDING_MODEL_TYPE in self.EMBEDDINGS_REGISTRY:
            supported_models = EMBEDDING_MODEL_TYPE.list_supported_models()
            if any(model_name.lower() == model["model"].lower() for model in supported_models):
                self.model = EMBEDDING_MODEL_TYPE(
                    model_name=model_name,
                    cache_dir=cache_dir,
                    threads=threads,
                    providers=providers,
                    cuda=cuda,
                    device_ids=device_ids,
                    lazy_load=lazy_load,
                    **kwargs,
                )
                return

        raise ValueError(
            f"Model {model_name} is not supported in TextEmbedding. "
            "Please check the supported models using `TextEmbedding.list_supported_models()`"
        )

    def embed(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs,
    ) -> Iterable[NDArray[np.float32]]:
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

    def passage_embed(self, texts: Iterable[str], **kwargs) -> Iterable[np.ndarray]:
        """
        Embeds a list of text passages into a list of embeddings.

        Args:
            texts (Iterable[str]): The list of texts to embed.
            **kwargs: Additional keyword argument to pass to the embed method.

        Yields:
            Iterable[SparseEmbedding]: The sparse embeddings.
        """
        # This is model-specific, so that different models can have specialized implementations
        yield from self.model.passage_embed(texts, **kwargs)
