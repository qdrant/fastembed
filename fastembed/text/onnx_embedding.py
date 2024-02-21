import os
from multiprocessing import get_all_start_methods
from typing import List, Dict, Any, Optional, Tuple, Union, Iterable, Type

import numpy as np
import onnxruntime as ort

from fastembed.common.model_management import locate_model_file
from fastembed.common.models import load_tokenizer, normalize
from fastembed.common.utils import define_cache_dir, iter_batch
from fastembed.parallel_processor import ParallelWorkerPool, Worker
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


class OnnxTextEmbedding(TextEmbeddingBase):
    """Implementation of the Flag Embedding model."""

    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        """Lists the supported models.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the model information.
        """
        return supported_onnx_models

    @classmethod
    def _get_model_description(cls, model_name: str) -> Dict[str, Any]:
        """
        Gets the model description from the model_name.

        Args:
            model_name (str): The name of the model.

        raises:
            ValueError: If the model_name is not supported.

        Returns:
            Dict[str, Any]: The model description.
        """
        for model in cls.list_supported_models():
            if model_name == model["model"]:
                return model

        raise ValueError(f"Model {model_name} is not supported in FlagEmbedding.")

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

        model_path = locate_model_file(self._model_dir, ["model.onnx", "model_optimized.onnx"])

        # List of Execution Providers: https://onnxruntime.ai/docs/execution-providers
        onnx_providers = ["CPUExecutionProvider"]

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if self.threads is not None:
            so.intra_op_num_threads = self.threads
            so.inter_op_num_threads = self.threads

        self.tokenizer = load_tokenizer(model_dir=self._model_dir, max_length=self._max_length)
        self.model = ort.InferenceSession(str(model_path), providers=onnx_providers, sess_options=so)

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
        is_small = False

        if isinstance(documents, str):
            documents = [documents]
            is_small = True

        if isinstance(documents, list):
            if len(documents) < batch_size:
                is_small = True

        if parallel == 0:
            parallel = os.cpu_count()

        if parallel is None or is_small:
            for batch in iter_batch(documents, batch_size):
                yield from self._post_process_onnx_output(self.onnx_embed(batch))
        else:
            start_method = "forkserver" if "forkserver" in get_all_start_methods() else "spawn"
            params = {
                "model_name": self.model_name,
                "cache_dir": str(self._cache_dir),
            }
            pool = ParallelWorkerPool(parallel, self._get_worker_class(), start_method=start_method)
            for batch in pool.ordered_map(iter_batch(documents, batch_size), **params):
                yield from self._post_process_onnx_output(batch)

    @classmethod
    def _get_worker_class(cls) -> Type["EmbeddingWorker"]:
        return OnnxTextEmbeddingWorker

    def _preprocess_onnx_input(self, onnx_input: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Preprocess the onnx input.
        """
        return onnx_input

    @classmethod
    def _post_process_onnx_output(cls, output: Tuple[np.ndarray, np.ndarray]):
        embeddings, _ = output
        return normalize(embeddings[:, 0]).astype(np.float32)

    def onnx_embed(self, documents: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        encoded = self.tokenizer.encode_batch(documents)
        input_ids = np.array([e.ids for e in encoded])
        attention_mask = np.array([e.attention_mask for e in encoded])

        onnx_input = {
            "input_ids": np.array(input_ids, dtype=np.int64),
            "attention_mask": np.array(attention_mask, dtype=np.int64),
            "token_type_ids": np.array([np.zeros(len(e), dtype=np.int64) for e in input_ids], dtype=np.int64),
        }

        onnx_input = self._preprocess_onnx_input(onnx_input)

        model_output = self.model.run(None, onnx_input)
        embeddings = model_output[0]
        return embeddings, attention_mask


class EmbeddingWorker(Worker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
    ) -> OnnxTextEmbedding:
        raise NotImplementedError()

    def __init__(
        self,
        model_name: str,
        cache_dir: str,
    ):
        self.model = self.init_embedding(model_name, cache_dir)

    @classmethod
    def start(cls, model_name: str, cache_dir: str, **kwargs: Any) -> "EmbeddingWorker":
        return cls(
            model_name=model_name,
            cache_dir=cache_dir,
        )

    def process(self, items: Iterable[Tuple[int, Any]]) -> Iterable[Tuple[int, Any]]:
        for idx, batch in items:
            embeddings, attn_mask = self.model.onnx_embed(batch)
            yield idx, (embeddings, attn_mask)


class OnnxTextEmbeddingWorker(EmbeddingWorker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
    ) -> OnnxTextEmbedding:
        return OnnxTextEmbedding(model_name=model_name, cache_dir=cache_dir, threads=1)
