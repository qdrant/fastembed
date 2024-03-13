from typing import Dict, Optional, Tuple, Union, Iterable, Type, List, Any

import numpy as np

from fastembed.common.onnx_model import OnnxModel, EmbeddingWorker
from fastembed.common.models import normalize
from fastembed.common.utils import define_cache_dir
from fastembed.text.text_embedding_base import TextEmbeddingBase

supported_onnx_models = [
    {
        "model": "BAAI/bge-base-en",
        "dim": 768,
        "description": "Base English model",
        "size_in_GB": 0.5,
        "sources": {
            "url": "https://storage.googleapis.com/qdrant-fastembed/fast-bge-base-en.tar.gz",
        },
    },
    {
        "model": "BAAI/bge-base-en-v1.5",
        "dim": 768,
        "description": "Base English model, v1.5",
        "size_in_GB": 0.44,
        "sources": {
            "url": "https://storage.googleapis.com/qdrant-fastembed/fast-bge-base-en-v1.5.tar.gz",
            "hf": "qdrant/bge-base-en-v1.5-onnx-q",
        },
    },
    {
        "model": "BAAI/bge-large-en-v1.5-quantized",
        "dim": 1024,
        "description": "Large English model, v1.5",
        "size_in_GB": 1.34,
        "sources": {
            "hf": "qdrant/bge-large-en-v1.5-onnx-q",
        },
    },
    {
        "model": "BAAI/bge-large-en-v1.5",
        "dim": 1024,
        "description": "Large English model, v1.5",
        "size_in_GB": 1.34,
        "sources": {
            "hf": "qdrant/bge-large-en-v1.5-onnx",
        },
    },
    {
        "model": "BAAI/bge-small-en",
        "dim": 384,
        "description": "Fast English model",
        "size_in_GB": 0.2,
        "sources": {
            "url": "https://storage.googleapis.com/qdrant-fastembed/BAAI-bge-small-en.tar.gz",
        },
    },
    # {
    #     "model": "BAAI/bge-small-en",
    #     "dim": 384,
    #     "description": "Fast English model",
    #     "size_in_GB": 0.2,
    #     "hf_sources": [],
    #     "compressed_url_sources": [
    #         "https://storage.googleapis.com/qdrant-fastembed/fast-bge-small-en.tar.gz",
    #         "https://storage.googleapis.com/qdrant-fastembed/BAAI-bge-small-en.tar.gz"
    #     ]
    # },
    {
        "model": "BAAI/bge-small-en-v1.5",
        "dim": 384,
        "description": "Fast and Default English model",
        "size_in_GB": 0.13,
        "sources": {
            "url": "https://storage.googleapis.com/qdrant-fastembed/fast-bge-small-en-v1.5.tar.gz",
            "hf": "qdrant/bge-small-en-v1.5-onnx-q",
        },
    },
    {
        "model": "BAAI/bge-small-zh-v1.5",
        "dim": 512,
        "description": "Fast and recommended Chinese model",
        "size_in_GB": 0.1,
        "sources": {
            "url": "https://storage.googleapis.com/qdrant-fastembed/fast-bge-small-zh-v1.5.tar.gz",
        },
    },
    {  # todo: it is not a flag embedding
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "description": "Sentence Transformer model, MiniLM-L6-v2",
        "size_in_GB": 0.09,
        "sources": {
            "url": "https://storage.googleapis.com/qdrant-fastembed/sentence-transformers-all-MiniLM-L6-v2.tar.gz",
            "hf": "qdrant/all-MiniLM-L6-v2-onnx",
        },
    },
    {
        "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "dim": 384,
        "description": "Sentence Transformer model, paraphrase-multilingual-MiniLM-L12-v2",
        "size_in_GB": 0.46,
        "sources": {
            "hf": "qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q",
        },
    },
    {
        "model": "nomic-ai/nomic-embed-text-v1",
        "dim": 768,
        "description": "8192 context length english model",
        "size_in_GB": 0.54,
        "sources": {
            "hf": "nomic-ai/nomic-embed-text-v1",
        },
    },
    {
        "model": "nomic-ai/nomic-embed-text-v1.5",
        "dim": 768,
        "description": "8192 context length english model",
        "size_in_GB": 0.54,
        "sources": {
            "hf": "nomic-ai/nomic-embed-text-v1.5",
        },
    },
    {
        "model": "thenlper/gte-large",
        "dim": 1024,
        "description": "Large general text embeddings model",
        "size_in_GB": 1.34,
        "sources": {
            "hf": "qdrant/gte-large-onnx",
        },
    },
    # {
    #     "model": "sentence-transformers/all-MiniLM-L6-v2",
    #     "dim": 384,
    #     "description": "Sentence Transformer model, MiniLM-L6-v2",
    #     "size_in_GB": 0.09,
    #     "hf_sources": [
    #         "qdrant/all-MiniLM-L6-v2-onnx"
    #     ],
    #     "compressed_url_sources": [
    #         "https://storage.googleapis.com/qdrant-fastembed/fast-all-MiniLM-L6-v2.tar.gz",
    #         "https://storage.googleapis.com/qdrant-fastembed/sentence-transformers-all-MiniLM-L6-v2.tar.gz"
    #     ]
    # }
]


class OnnxTextEmbedding(TextEmbeddingBase, OnnxModel[np.ndarray]):
    """Implementation of the Flag Embedding model."""

    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        """
        Lists the supported models.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the model information.
        """
        return supported_onnx_models

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            model_name (str): The name of the model to use.
            cache_dir (str, optional): The path to the cache directory.
                                       Can be set using the `FASTEMBED_CACHE_PATH` env variable.
                                       Defaults to `fastembed_cache` in the system's temp directory.
            threads (int, optional): The number of threads single onnxruntime session can use. Defaults to None.

        Raises:
            ValueError: If the model_name is not in the format <org>/<model> e.g. BAAI/bge-base-en.
        """

        super().__init__(model_name, cache_dir, threads, **kwargs)

        self.model_name = model_name
        self._model_description = self._get_model_description(model_name)

        self._cache_dir = define_cache_dir(cache_dir)
        self._model_dir = self.download_model(self._model_description, self._cache_dir)
        self._max_length = 512

        self.load_onnx_model(self._model_dir, self.threads, self._max_length)

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
            cache_dir=str(self._cache_dir),
            documents=documents,
            batch_size=batch_size,
            parallel=parallel,
        )

    @classmethod
    def _get_worker_class(cls) -> Type["EmbeddingWorker"]:
        return OnnxTextEmbeddingWorker

    def _preprocess_onnx_input(self, onnx_input: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Preprocess the onnx input.
        """
        return onnx_input

    @classmethod
    def _post_process_onnx_output(cls, output: Tuple[np.ndarray, np.ndarray]) -> Iterable[np.ndarray]:
        embeddings, _ = output
        return normalize(embeddings[:, 0]).astype(np.float32)


class OnnxTextEmbeddingWorker(EmbeddingWorker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
    ) -> OnnxTextEmbedding:
        return OnnxTextEmbedding(model_name=model_name, cache_dir=cache_dir, threads=1)
