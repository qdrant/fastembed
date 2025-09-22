from typing import Any, Iterable, Optional, Sequence, Type, Union

from fastembed.common.types import NumpyArray, OnnxProvider
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.utils import define_cache_dir, normalize
from fastembed.text.onnx_text_model import OnnxTextModel, TextEmbeddingWorker
from fastembed.text.text_embedding_base import TextEmbeddingBase
from fastembed.common.model_description import DenseModelDescription, ModelSource

supported_onnx_models: list[DenseModelDescription] = [
    DenseModelDescription(
        model="BAAI/bge-base-en",
        dim=768,
        description=(
            "Text embeddings, Unimodal (text), English, 512 input tokens truncation, "
            "Prefixes for queries/documents: necessary, 2023 year."
        ),
        license="mit",
        size_in_GB=0.42,
        sources=ModelSource(
            hf="Qdrant/fast-bge-base-en",
            url="https://storage.googleapis.com/qdrant-fastembed/fast-bge-base-en.tar.gz",
            _deprecated_tar_struct=True,
        ),
        model_file="model_optimized.onnx",
    ),
    DenseModelDescription(
        model="BAAI/bge-base-en-v1.5",
        dim=768,
        description=(
            "Text embeddings, Unimodal (text), English, 512 input tokens truncation, "
            "Prefixes for queries/documents: not so necessary, 2023 year."
        ),
        license="mit",
        size_in_GB=0.21,
        sources=ModelSource(
            hf="qdrant/bge-base-en-v1.5-onnx-q",
            url="https://storage.googleapis.com/qdrant-fastembed/fast-bge-base-en-v1.5.tar.gz",
            _deprecated_tar_struct=True,
        ),
        model_file="model_optimized.onnx",
    ),
    DenseModelDescription(
        model="BAAI/bge-large-en-v1.5",
        dim=1024,
        description=(
            "Text embeddings, Unimodal (text), English, 512 input tokens truncation, "
            "Prefixes for queries/documents: not so necessary, 2023 year."
        ),
        license="mit",
        size_in_GB=1.20,
        sources=ModelSource(hf="qdrant/bge-large-en-v1.5-onnx"),
        model_file="model.onnx",
    ),
    DenseModelDescription(
        model="BAAI/bge-small-en",
        dim=384,
        description=(
            "Text embeddings, Unimodal (text), English, 512 input tokens truncation, "
            "Prefixes for queries/documents: necessary, 2023 year."
        ),
        license="mit",
        size_in_GB=0.13,
        sources=ModelSource(
            hf="Qdrant/bge-small-en",
            url="https://storage.googleapis.com/qdrant-fastembed/BAAI-bge-small-en.tar.gz",
            _deprecated_tar_struct=True,
        ),
        model_file="model_optimized.onnx",
    ),
    DenseModelDescription(
        model="BAAI/bge-small-en-v1.5",
        dim=384,
        description=(
            "Text embeddings, Unimodal (text), English, 512 input tokens truncation, "
            "Prefixes for queries/documents: not so necessary, 2023 year."
        ),
        license="mit",
        size_in_GB=0.067,
        sources=ModelSource(hf="qdrant/bge-small-en-v1.5-onnx-q"),
        model_file="model_optimized.onnx",
    ),
    DenseModelDescription(
        model="BAAI/bge-small-zh-v1.5",
        dim=512,
        description=(
            "Text embeddings, Unimodal (text), Chinese, 512 input tokens truncation, "
            "Prefixes for queries/documents: not so necessary, 2023 year."
        ),
        license="mit",
        size_in_GB=0.09,
        sources=ModelSource(
            hf="Qdrant/bge-small-zh-v1.5",
            url="https://storage.googleapis.com/qdrant-fastembed/fast-bge-small-zh-v1.5.tar.gz",
            _deprecated_tar_struct=True,
        ),
        model_file="model_optimized.onnx",
    ),
    DenseModelDescription(
        model="mixedbread-ai/mxbai-embed-large-v1",
        dim=1024,
        description=(
            "Text embeddings, Unimodal (text), English, 512 input tokens truncation, "
            "Prefixes for queries/documents: necessary, 2024 year."
        ),
        license="apache-2.0",
        size_in_GB=0.64,
        sources=ModelSource(hf="mixedbread-ai/mxbai-embed-large-v1"),
        model_file="onnx/model.onnx",
    ),
    DenseModelDescription(
        model="snowflake/snowflake-arctic-embed-xs",
        dim=384,
        description=(
            "Text embeddings, Unimodal (text), English, 512 input tokens truncation, "
            "Prefixes for queries/documents: necessary, 2024 year."
        ),
        license="apache-2.0",
        size_in_GB=0.09,
        sources=ModelSource(hf="snowflake/snowflake-arctic-embed-xs"),
        model_file="onnx/model.onnx",
    ),
    DenseModelDescription(
        model="snowflake/snowflake-arctic-embed-s",
        dim=384,
        description=(
            "Text embeddings, Unimodal (text), English, 512 input tokens truncation, "
            "Prefixes for queries/documents: necessary, 2024 year."
        ),
        license="apache-2.0",
        size_in_GB=0.13,
        sources=ModelSource(hf="snowflake/snowflake-arctic-embed-s"),
        model_file="onnx/model.onnx",
    ),
    DenseModelDescription(
        model="snowflake/snowflake-arctic-embed-m",
        dim=768,
        description=(
            "Text embeddings, Unimodal (text), English, 512 input tokens truncation, "
            "Prefixes for queries/documents: necessary, 2024 year."
        ),
        license="apache-2.0",
        size_in_GB=0.43,
        sources=ModelSource(hf="Snowflake/snowflake-arctic-embed-m"),
        model_file="onnx/model.onnx",
    ),
    DenseModelDescription(
        model="snowflake/snowflake-arctic-embed-m-long",
        dim=768,
        description=(
            "Text embeddings, Unimodal (text), English, 2048 input tokens truncation, "
            "Prefixes for queries/documents: necessary, 2024 year."
        ),
        license="apache-2.0",
        size_in_GB=0.54,
        sources=ModelSource(hf="snowflake/snowflake-arctic-embed-m-long"),
        model_file="onnx/model.onnx",
    ),
    DenseModelDescription(
        model="snowflake/snowflake-arctic-embed-l",
        dim=1024,
        description=(
            "Text embeddings, Unimodal (text), English, 512 input tokens truncation, "
            "Prefixes for queries/documents: necessary, 2024 year."
        ),
        license="apache-2.0",
        size_in_GB=1.02,
        sources=ModelSource(hf="snowflake/snowflake-arctic-embed-l"),
        model_file="onnx/model.onnx",
    ),
    DenseModelDescription(
        model="jinaai/jina-clip-v1",
        dim=768,
        description=(
            "Text embeddings, Multimodal (text&image), English, Prefixes for queries/documents: "
            "not necessary, 2024 year"
        ),
        license="apache-2.0",
        size_in_GB=0.55,
        sources=ModelSource(hf="jinaai/jina-clip-v1"),
        model_file="onnx/text_model.onnx",
    ),
]


class OnnxTextEmbedding(TextEmbeddingBase, OnnxTextModel[NumpyArray]):
    """Implementation of the Flag Embedding model."""

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        """
        Lists the supported models.

        Returns:
            list[DenseModelDescription]: A list of DenseModelDescription objects containing the model information.
        """
        return supported_onnx_models

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence[OnnxProvider]] = None,
        cuda: bool = False,
        device_ids: Optional[list[int]] = None,
        lazy_load: bool = False,
        device_id: Optional[int] = None,
        specific_model_path: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initializes an ONNX-based text embedding model with configurable device, threading, and model source options.
        
        Args:
            model_name: Name of the ONNX model to use, in the format <org>/<model>.
            cache_dir: Directory for caching downloaded models. If not provided, defaults to a system temp directory.
            threads: Number of threads for ONNX runtime session.
            providers: Sequence of ONNX runtime providers to use. Mutually exclusive with `cuda` and `device_ids`.
            cuda: Whether to use CUDA for inference. Mutually exclusive with `providers`.
            device_ids: List of device IDs for data parallel processing. Should be used with `cuda=True`.
            lazy_load: If True, delays model loading until first use, useful for multi-GPU or parallel encoding.
            device_id: Device ID for loading the model in the current process.
            specific_model_path: Path to a specific ONNX model directory to load from an external location.
        
        Raises:
            ValueError: If `model_name` is not in the required <org>/<model> format.
        """
        super().__init__(model_name, cache_dir, threads, **kwargs)
        self.providers = providers
        self.lazy_load = lazy_load

        # List of device ids, that can be used for data parallel processing in workers
        self.device_ids = device_ids
        self.cuda = cuda

        # This device_id will be used if we need to load model in current process
        self.device_id: Optional[int] = None
        if device_id is not None:
            self.device_id = device_id
        elif self.device_ids is not None:
            self.device_id = self.device_ids[0]

        self.model_description = self._get_model_description(model_name)
        self.cache_dir = str(define_cache_dir(cache_dir))
        self._specific_model_path = specific_model_path
        self._model_dir = self.download_model(
            self.model_description,
            self.cache_dir,
            local_files_only=self._local_files_only,
            specific_model_path=self._specific_model_path,
        )

        if not self.lazy_load:
            self.load_onnx_model()

    def embed(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        """
        Generates embeddings for one or more documents using the ONNX text embedding model.
        
        Args:
            documents: A string or iterable of strings representing the documents to embed.
            batch_size: Number of documents to process per batch. Larger values may improve speed at the cost of memory usage.
            parallel: Number of parallel workers to use for data-parallel encoding. If 0, uses all available cores. If None, disables parallel processing.
        
        Returns:
            An iterable of numpy arrays, each representing the embedding of a document.
        """
        yield from self._embed_documents(
            model_name=self.model_name,
            cache_dir=str(self.cache_dir),
            documents=documents,
            batch_size=batch_size,
            parallel=parallel,
            providers=self.providers,
            cuda=self.cuda,
            device_ids=self.device_ids,
            local_files_only=self._local_files_only,
            specific_model_path=self._specific_model_path,
            **kwargs,
        )

    @classmethod
    def _get_worker_class(cls) -> Type["TextEmbeddingWorker[NumpyArray]"]:
        return OnnxTextEmbeddingWorker

    def _preprocess_onnx_input(
        self, onnx_input: dict[str, NumpyArray], **kwargs: Any
    ) -> dict[str, NumpyArray]:
        """
        Preprocess the onnx input.
        """
        return onnx_input

    def _post_process_onnx_output(
        self, output: OnnxOutputContext, **kwargs: Any
    ) -> Iterable[NumpyArray]:
        embeddings = output.model_output

        if embeddings.ndim == 3:  # (batch_size, seq_len, embedding_dim)
            processed_embeddings = embeddings[:, 0]
        elif embeddings.ndim == 2:  # (batch_size, embedding_dim)
            processed_embeddings = embeddings
        else:
            raise ValueError(f"Unsupported embedding shape: {embeddings.shape}")
        return normalize(processed_embeddings)

    def load_onnx_model(self) -> None:
        self._load_onnx_model(
            model_dir=self._model_dir,
            model_file=self.model_description.model_file,
            threads=self.threads,
            providers=self.providers,
            cuda=self.cuda,
            device_id=self.device_id,
        )


class OnnxTextEmbeddingWorker(TextEmbeddingWorker[NumpyArray]):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs: Any,
    ) -> OnnxTextEmbedding:
        return OnnxTextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
