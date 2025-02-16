import warnings
from typing import Any, Iterable, Optional, Sequence, Type, Union
from dataclasses import asdict

from fastembed.common.types import NumpyArray, OnnxProvider
from fastembed.text.clip_embedding import CLIPOnnxEmbedding
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
        tasks: Optional[dict[str, Any]] = None,
    ) -> None:
        registered_models = cls._list_supported_models()
        for registered_model in registered_models:
            if model == registered_model.model:
                raise ValueError(
                    f"Model {model} is already registered in TextEmbedding, if you still want to add this model, "
                    f"please use another model name"
                )

        if tasks:
            if pooling == PoolingType.MEAN and normalization:
                JinaEmbeddingV3.add_custom_model(
                    model=model,
                    sources=sources,
                    dim=dim,
                    model_file=model_file,
                    description=description,
                    license=license,
                    size_in_gb=size_in_gb,
                    additional_files=additional_files,
                    tasks=tasks,
                )
                return None
            else:
                raise ValueError(
                    "Multitask models supported only with pooling=Pooling.MEAN and normalization=True, current values:"
                    f"pooling={pooling}, normalization={normalization}, tasks: {tasks}"
                )

        embedding_cls: Type[OnnxTextEmbedding]
        if pooling == PoolingType.MEAN and normalization:
            embedding_cls = PooledNormalizedEmbedding
        elif pooling == PoolingType.MEAN and not normalization:
            embedding_cls = PooledEmbedding
        elif (pooling == PoolingType.CLS or PoolingType.DISABLED) and normalization:
            embedding_cls = OnnxTextEmbedding
        elif pooling == PoolingType.DISABLED and not normalization:
            embedding_cls = CLIPOnnxEmbedding
        else:
            raise ValueError(
                "Only the following combinations of pooling and normalization are currently supported:"
                "pooling=Pooling.MEAN + normalization=True;\n"
                "pooling=Pooling.MEAN + normalization=False;\n"
                "pooling=Pooling.CLS + normalization=True;\n"
                "pooling=Pooling.DISABLED + normalization=False;\n"
            )

        embedding_cls.add_custom_model(
            model=model,
            sources=sources,
            dim=dim,
            model_file=model_file,
            description=description,
            license=license,
            size_in_gb=size_in_gb,
            additional_files=additional_files,
            tasks=tasks,
        )
        return None

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
        if model_name in {
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "intfloat/multilingual-e5-large",
        }:
            warnings.warn(
                f"{model_name} has been updated as of fastembed 0.5.2, outputs are now average pooled.",
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
