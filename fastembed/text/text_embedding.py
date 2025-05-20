import warnings
from typing import Any, Iterable, Optional, Sequence, Type, Union
from dataclasses import asdict

from fastembed.common.types import NumpyArray, OnnxProvider
from fastembed.text.clip_embedding import CLIPOnnxEmbedding
from fastembed.text.custom_text_embedding import CustomTextEmbedding
from fastembed.text.pooled_normalized_embedding import PooledNormalizedEmbedding
from fastembed.text.pooled_embedding import PooledEmbedding
from fastembed.text.multitask_embedding import JinaEmbeddingV3
from fastembed.text.onnx_embedding import OnnxTextEmbedding
from fastembed.text.text_embedding_base import TextEmbeddingBase
from fastembed.common.model_description import DenseModelDescription, ModelSource, PoolingType


class TextEmbedding(TextEmbeddingBase):
    EMBEDDINGS_REGISTRY: list[Type[TextEmbeddingBase]] = [
        OnnxTextEmbedding,
        CLIPOnnxEmbedding,
        PooledNormalizedEmbedding,
        PooledEmbedding,
        JinaEmbeddingV3,
        CustomTextEmbedding,
    ]

    @classmethod
    def list_supported_models(cls) -> list[dict[str, Any]]:
        """Lists the supported models.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing the model information.
        """
        return [asdict(model) for model in cls._list_supported_models()]

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        result: list[DenseModelDescription] = []
        for embedding in cls.EMBEDDINGS_REGISTRY:
            result.extend(embedding._list_supported_models())
        return result

    @classmethod
    def add_custom_model(
        cls,
        model: str,
        pooling: PoolingType,
        normalization: bool,
        sources: ModelSource,
        dim: int,
        model_file: str = "onnx/model.onnx",
        description: str = "",
        license: str = "",
        size_in_gb: float = 0.0,
        additional_files: Optional[list[str]] = None,
    ) -> None:
        registered_models = cls._list_supported_models()
        for registered_model in registered_models:
            if model.lower() == registered_model.model.lower():
                raise ValueError(
                    f"Model {model} is already registered in TextEmbedding, if you still want to add this model, "
                    f"please use another model name"
                )

        CustomTextEmbedding.add_model(
            DenseModelDescription(
                model=model,
                sources=sources,
                dim=dim,
                model_file=model_file,
                description=description,
                license=license,
                size_in_GB=size_in_gb,
                additional_files=additional_files or [],
            ),
            pooling=pooling,
            normalization=normalization,
        )

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence[OnnxProvider]] = None,
        cuda: bool = False,
        device_ids: Optional[list[int]] = None,
        lazy_load: bool = False,
        **kwargs: Any,
    ):
        super().__init__(model_name, cache_dir, threads, **kwargs)
        if model_name.lower() == "nomic-ai/nomic-embed-text-v1.5-Q".lower():
            warnings.warn(
                "The model 'nomic-ai/nomic-embed-text-v1.5-Q' has been updated on HuggingFace. Please review "
                "the latest documentation on HF and release notes to ensure compatibility with your workflow. ",
                UserWarning,
                stacklevel=2,
            )
        if model_name.lower() in {
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2".lower(),
            "thenlper/gte-large".lower(),
            "intfloat/multilingual-e5-large".lower(),
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2".lower(),
        }:
            warnings.warn(
                f"The model {model_name} now uses mean pooling instead of CLS embedding. "
                f"In order to preserve the previous behaviour, consider either pinning fastembed version to 0.5.1 or "
                "using `add_custom_model` functionality.",
                UserWarning,
                stacklevel=2,
            )
        for EMBEDDING_MODEL_TYPE in self.EMBEDDINGS_REGISTRY:
            supported_models = EMBEDDING_MODEL_TYPE._list_supported_models()
            if any(model_name.lower() == model.model.lower() for model in supported_models):
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

    @property
    def embedding_size(self) -> int:
        """Get the embedding size of the current model"""
        if self._embedding_size is None:
            self._embedding_size = self.get_embedding_size(self.model_name)
        return self._embedding_size

    @classmethod
    def get_embedding_size(cls, model_name: str) -> int:
        """Get the embedding size of the passed model

        Args:
            model_name (str): The name of the model to get embedding size for.

        Returns:
            int: The size of the embedding.

        Raises:
            ValueError: If the model name is not found in the supported models.
        """
        descriptions = cls._list_supported_models()
        embedding_size: Optional[int] = None
        for description in descriptions:
            if description.model.lower() == model_name.lower():
                embedding_size = description.dim
                break
        if embedding_size is None:
            model_names = [description.model for description in descriptions]
            raise ValueError(
                f"Embedding size for model {model_name} was None. "
                f"Available model names: {model_names}"
            )
        return embedding_size

    def embed(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
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

    def query_embed(self, query: Union[str, Iterable[str]], **kwargs: Any) -> Iterable[NumpyArray]:
        """
        Embeds queries

        Args:
            query (Union[str, Iterable[str]]): The query to embed, or an iterable e.g. list of queries.

        Returns:
            Iterable[NumpyArray]: The embeddings.
        """
        # This is model-specific, so that different models can have specialized implementations
        yield from self.model.query_embed(query, **kwargs)

    def passage_embed(self, texts: Iterable[str], **kwargs: Any) -> Iterable[NumpyArray]:
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
