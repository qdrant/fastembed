from typing import Any, Iterable, Optional, Sequence, Type, Union

import numpy as np

from fastembed.common import OnnxProvider
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.utils import define_cache_dir, normalize
from fastembed.text.onnx_text_model import OnnxTextModel, TextEmbeddingWorker
from fastembed.text.text_embedding_base import TextEmbeddingBase

supported_onnx_models = [
    {
        "model": "BAAI/bge-base-en",
        "dim": 768,
        "description": "Text embeddings, Unimodal (text), English, 512 input tokens truncation, Prefixes for queries/documents: necessary, 2023 year.",
        "license": "mit",
        "size_in_GB": 0.42,
        "sources": {
            "hf": "Qdrant/fast-bge-base-en",
            "url": "https://storage.googleapis.com/qdrant-fastembed/fast-bge-base-en.tar.gz",
        },
        "model_file": "model_optimized.onnx",
    },
    {
        "model": "BAAI/bge-base-en-v1.5",
        "dim": 768,
        "description": "Text embeddings, Unimodal (text), English, 512 input tokens truncation, Prefixes for queries/documents: not so necessary, 2023 year.",
        "license": "mit",
        "size_in_GB": 0.21,
        "sources": {
            "url": "https://storage.googleapis.com/qdrant-fastembed/fast-bge-base-en-v1.5.tar.gz",
            "hf": "qdrant/bge-base-en-v1.5-onnx-q",
        },
        "model_file": "model_optimized.onnx",
    },
    {
        "model": "BAAI/bge-large-en-v1.5",
        "dim": 1024,
        "description": "Text embeddings, Unimodal (text), English, 512 input tokens truncation, Prefixes for queries/documents: not so necessary, 2023 year.",
        "license": "mit",
        "size_in_GB": 1.20,
        "sources": {
            "hf": "qdrant/bge-large-en-v1.5-onnx",
        },
        "model_file": "model.onnx",
    },
    {
        "model": "BAAI/bge-small-en",
        "dim": 384,
        "description": "Text embeddings, Unimodal (text), English, 512 input tokens truncation, Prefixes for queries/documents: necessary, 2023 year.",
        "license": "mit",
        "size_in_GB": 0.13,
        "sources": {
            "hf": "Qdrant/bge-small-en",
            "url": "https://storage.googleapis.com/qdrant-fastembed/BAAI-bge-small-en.tar.gz",
        },
        "model_file": "model_optimized.onnx",
    },
    {
        "model": "BAAI/bge-small-en-v1.5",
        "dim": 384,
        "description": "Text embeddings, Unimodal (text), English, 512 input tokens truncation, Prefixes for queries/documents: not so necessary, 2023 year.",
        "license": "mit",
        "size_in_GB": 0.067,
        "sources": {
            "hf": "qdrant/bge-small-en-v1.5-onnx-q",
        },
        "model_file": "model_optimized.onnx",
    },
    {
        "model": "BAAI/bge-small-zh-v1.5",
        "dim": 512,
        "description": "Text embeddings, Unimodal (text), Chinese, 512 input tokens truncation, Prefixes for queries/documents: not so necessary, 2023 year.",
        "license": "mit",
        "size_in_GB": 0.09,
        "sources": {
            "hf": "Qdrant/bge-small-zh-v1.5",
            "url": "https://storage.googleapis.com/qdrant-fastembed/fast-bge-small-zh-v1.5.tar.gz",
        },
        "model_file": "model_optimized.onnx",
    },
    {
        "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "dim": 384,
        "description": "Text embeddings, Unimodal (text), Multilingual (~50 languages), 512 input tokens truncation, Prefixes for queries/documents: not necessary, 2019 year.",
        "license": "apache-2.0",
        "size_in_GB": 0.22,
        "sources": {
            "hf": "qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q",
        },
        "model_file": "model_optimized.onnx",
    },
    {
        "model": "thenlper/gte-large",
        "dim": 1024,
        "description": "Text embeddings, Unimodal (text), English, 512 input tokens truncation, Prefixes for queries/documents: not necessary, 2023 year.",
        "license": "mit",
        "size_in_GB": 1.20,
        "sources": {
            "hf": "qdrant/gte-large-onnx",
        },
        "model_file": "model.onnx",
    },
    {
        "model": "mixedbread-ai/mxbai-embed-large-v1",
        "dim": 1024,
        "description": "Text embeddings, Unimodal (text), English, 512 input tokens truncation, Prefixes for queries/documents: necessary, 2024 year.",
        "license": "apache-2.0",
        "size_in_GB": 0.64,
        "sources": {
            "hf": "mixedbread-ai/mxbai-embed-large-v1",
        },
        "model_file": "onnx/model.onnx",
    },
    {
        "model": "snowflake/snowflake-arctic-embed-xs",
        "dim": 384,
        "description": "Text embeddings, Unimodal (text), English, 512 input tokens truncation, Prefixes for queries/documents: necessary, 2024 year.",
        "license": "apache-2.0",
        "size_in_GB": 0.09,
        "sources": {
            "hf": "snowflake/snowflake-arctic-embed-xs",
        },
        "model_file": "onnx/model.onnx",
    },
    {
        "model": "snowflake/snowflake-arctic-embed-s",
        "dim": 384,
        "description": "Text embeddings, Unimodal (text), English, 512 input tokens truncation, Prefixes for queries/documents: necessary, 2024 year.",
        "license": "apache-2.0",
        "size_in_GB": 0.13,
        "sources": {
            "hf": "snowflake/snowflake-arctic-embed-s",
        },
        "model_file": "onnx/model.onnx",
    },
    {
        "model": "snowflake/snowflake-arctic-embed-m",
        "dim": 768,
        "description": "Text embeddings, Unimodal (text), English, 512 input tokens truncation, Prefixes for queries/documents: necessary, 2024 year.",
        "license": "apache-2.0",
        "size_in_GB": 0.43,
        "sources": {
            "hf": "Snowflake/snowflake-arctic-embed-m",
        },
        "model_file": "onnx/model.onnx",
    },
    {
        "model": "snowflake/snowflake-arctic-embed-m-long",
        "dim": 768,
        "description": "Text embeddings, Unimodal (text), English, 2048 input tokens truncation, Prefixes for queries/documents: necessary, 2024 year.",
        "license": "apache-2.0",
        "size_in_GB": 0.54,
        "sources": {
            "hf": "snowflake/snowflake-arctic-embed-m-long",
        },
        "model_file": "onnx/model.onnx",
    },
    {
        "model": "snowflake/snowflake-arctic-embed-l",
        "dim": 1024,
        "description": "Text embeddings, Unimodal (text), English, 512 input tokens truncation, Prefixes for queries/documents: necessary, 2024 year.",
        "license": "apache-2.0",
        "size_in_GB": 1.02,
        "sources": {
            "hf": "snowflake/snowflake-arctic-embed-l",
        },
        "model_file": "onnx/model.onnx",
    },
    {
        "model": "jinaai/jina-clip-v1",
        "dim": 768,
        "description": "Text embeddings, Multimodal (text&image), English, Prefixes for queries/documents: not necessary, 2024 year",
        "license": "apache-2.0",
        "size_in_GB": 0.55,
        "sources": {
            "hf": "jinaai/jina-clip-v1",
        },
        "model_file": "onnx/text_model.onnx",
    },
]


class OnnxTextEmbedding(TextEmbeddingBase, OnnxTextModel[np.ndarray]):
    """Implementation of the Flag Embedding model."""

    @classmethod
    def list_supported_models(cls) -> list[dict[str, Any]]:
        """
        Lists the supported models.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing the model information.
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
        **kwargs,
    ):
        """
        Args:
            model_name (str): The name of the model to use.
            cache_dir (str, optional): The path to the cache directory.
                                       Can be set using the `FASTEMBED_CACHE_PATH` env variable.
                                       Defaults to `fastembed_cache` in the system's temp directory.
            threads (int, optional): The number of threads single onnxruntime session can use. Defaults to None.
            providers (Optional[Sequence[OnnxProvider]], optional): The list of onnxruntime providers to use.
                Mutually exclusive with the `cuda` and `device_ids` arguments. Defaults to None.
            cuda (bool, optional): Whether to use cuda for inference. Mutually exclusive with `providers`
                Defaults to False.
            device_ids (Optional[list[int]], optional): The list of device ids to use for data parallel processing in
                workers. Should be used with `cuda=True`, mutually exclusive with `providers`. Defaults to None.
            lazy_load (bool, optional): Whether to load the model during class initialization or on demand.
                Should be set to True when using multiple-gpu and parallel encoding. Defaults to False.
            device_id (Optional[int], optional): The device id to use for loading the model in the worker process.

        Raises:
            ValueError: If the model_name is not in the format <org>/<model> e.g. BAAI/bge-base-en.
        """
        super().__init__(model_name, cache_dir, threads, **kwargs)
        self.providers = providers
        self.lazy_load = lazy_load

        # List of device ids, that can be used for data parallel processing in workers
        self.device_ids = device_ids
        self.cuda = cuda

        # This device_id will be used if we need to load model in current process
        if device_id is not None:
            self.device_id = device_id
        elif self.device_ids is not None:
            self.device_id = self.device_ids[0]
        else:
            self.device_id = None

        self.model_description = self._get_model_description(model_name)
        self.cache_dir = define_cache_dir(cache_dir)
        self._model_dir = self.download_model(
            self.model_description, self.cache_dir, local_files_only=self._local_files_only
        )

        if not self.lazy_load:
            self.load_onnx_model()

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
        yield from self._embed_documents(
            model_name=self.model_name,
            cache_dir=str(self.cache_dir),
            documents=documents,
            batch_size=batch_size,
            parallel=parallel,
            providers=self.providers,
            cuda=self.cuda,
            device_ids=self.device_ids,
            **kwargs,
        )

    @classmethod
    def _get_worker_class(cls) -> Type["TextEmbeddingWorker"]:
        return OnnxTextEmbeddingWorker

    def _preprocess_onnx_input(
        self, onnx_input: dict[str, np.ndarray], **kwargs
    ) -> dict[str, np.ndarray]:
        """
        Preprocess the onnx input.
        """
        return onnx_input

    def _post_process_onnx_output(self, output: OnnxOutputContext) -> Iterable[np.ndarray]:
        embeddings = output.model_output
        if embeddings.ndim == 3:  # (batch_size, seq_len, embedding_dim)
            processed_embeddings = embeddings[:, 0]
        elif embeddings.ndim == 2:  # (batch_size, embedding_dim)
            processed_embeddings = embeddings
        else:
            raise ValueError(f"Unsupported embedding shape: {embeddings.shape}")
        return normalize(processed_embeddings).astype(np.float32)

    def load_onnx_model(self) -> None:
        self._load_onnx_model(
            model_dir=self._model_dir,
            model_file=self.model_description["model_file"],
            threads=self.threads,
            providers=self.providers,
            cuda=self.cuda,
            device_id=self.device_id,
        )


class OnnxTextEmbeddingWorker(TextEmbeddingWorker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs,
    ) -> OnnxTextEmbedding:
        return OnnxTextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
