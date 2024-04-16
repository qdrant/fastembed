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
        "size_in_GB": 0.42,
        "sources": {
            "hf": "yashvardhan7/bge-base-en-onnx",
        },
        "model_file": "onnx/model.onnx",
    },
    {
        "model": "BAAI/bge-base-en-v1.5",
        "dim": 768,
        "description": "Base English model, v1.5",
        "size_in_GB": 0.21,
        "sources": {
            "hf": "qdrant/bge-base-en-v1.5-onnx-q",
        },
        "model_file": "model_optimized.onnx",
    },
    {
        "model": "BAAI/bge-large-en-v1.5",
        "dim": 1024,
        "description": "Large English model, v1.5",
        "size_in_GB": 1.20,
        "sources": {
            "hf": "qdrant/bge-large-en-v1.5-onnx",
        },
        "model_file": "model.onnx",
    },
    {
        "model": "BAAI/bge-small-en",
        "dim": 384,
        "description": "Fast English model",
        "size_in_GB": 0.13,
        "sources": {
            "hf": "ggrn/bge-small-en",
        },
        "model_file": "onnx/model.onnx",
    },
    {
        "model": "BAAI/bge-small-en-v1.5",
        "dim": 384,
        "description": "Fast and Default English model",
        "size_in_GB": 0.067,
        "sources": {
            "hf": "qdrant/bge-small-en-v1.5-onnx-q",
        },
        "model_file": "Qdrant/bge-small-en-v1.5-onnx-Q",
    },
    {
        "model": "BAAI/bge-small-zh-v1.5",
        "dim": 512,
        "description": "Fast and recommended Chinese model",
        "size_in_GB": 0.09,
        "sources": {
            "hf": "Xenova/bge-small-zh-v1.5",
        },
        "model_file": "onnx/model.onnx",
    },
    {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "description": "Sentence Transformer model, MiniLM-L6-v2",
        "size_in_GB": 0.09,
        "sources": {
            "hf": "qdrant/all-MiniLM-L6-v2-onnx",
        },
        "model_file": "model.onnx",
    },
    {
        "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "dim": 384,
        "description": "Sentence Transformer model, paraphrase-multilingual-MiniLM-L12-v2",
        "size_in_GB": 0.22,
        "sources": {
            "hf": "qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q",
        },
        "model_file": "model_optimized.onnx",
    },
    {
        "model": "nomic-ai/nomic-embed-text-v1",
        "dim": 768,
        "description": "8192 context length english model",
        "size_in_GB": 0.52,
        "sources": {
            "hf": "nomic-ai/nomic-embed-text-v1",
        },
        "model_file": "onnx/model.onnx",
    },
    {
        "model": "nomic-ai/nomic-embed-text-v1.5",
        "dim": 768,
        "description": "8192 context length english model",
        "size_in_GB": 0.52,
        "sources": {
            "hf": "nomic-ai/nomic-embed-text-v1.5",
        },
        "model_file": "onnx/model.onnx",
    },
    {
        "model": "nomic-ai/nomic-embed-text-v1.5-Q",
        "dim": 768,
        "description": "Quantized 8192 context length english model",
        "size_in_GB": 0.13,
        "sources": {
            "hf": "nomic-ai/nomic-embed-text-v1.5",
        },
        "model_file": "onnx/model_quantized.onnx",
    },
    {
        "model": "thenlper/gte-large",
        "dim": 1024,
        "description": "Large general text embeddings model",
        "size_in_GB": 1.20,
        "sources": {
            "hf": "qdrant/gte-large-onnx",
        },
        "model_file": "model.onnx",
    },
    {
        "model": "mixedbread-ai/mxbai-embed-large-v1",
        "dim": 1024,
        "description": "MixedBread Base sentence embedding model, does well on MTEB",
        "size_in_GB": 0.64,
        "sources": {
            "hf": "mixedbread-ai/mxbai-embed-large-v1",
        },
        "model_file": "onnx/model.onnx",
    },
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

        self.load_onnx_model(
            self._get_model_description(model_name),
            threads,
            define_cache_dir(cache_dir),
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
        yield from self._embed_documents(
            model_name=self.model_name,
            cache_dir=str(self.cache_dir),
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
    def _post_process_onnx_output(
        cls, output: Tuple[np.ndarray, np.ndarray]
    ) -> Iterable[np.ndarray]:
        embeddings, _ = output
        return normalize(embeddings[:, 0]).astype(np.float32)


class OnnxTextEmbeddingWorker(EmbeddingWorker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
    ) -> OnnxTextEmbedding:
        return OnnxTextEmbedding(model_name=model_name, cache_dir=cache_dir, threads=1)
