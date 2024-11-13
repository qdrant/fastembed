from typing import List, Iterable, Dict, Any, Sequence, Optional

from loguru import logger

from fastembed.common import OnnxProvider
from fastembed.rerank.cross_encoder.onnx_text_model import OnnxCrossEncoderModel
from fastembed.rerank.cross_encoder.text_cross_encoder_base import TextCrossEncoderBase
from fastembed.common.utils import define_cache_dir

supported_onnx_models = [
    {
        "model": "Xenova/ms-marco-MiniLM-L-6-v2",
        "size_in_GB": 0.08,
        "sources": {
            "hf": "Xenova/ms-marco-MiniLM-L-6-v2",
        },
        "model_file": "onnx/model.onnx",
        "description": "MiniLM-L-6-v2 model optimized for re-ranking tasks.",
        "license": "apache-2.0",
    },
    {
        "model": "Xenova/ms-marco-MiniLM-L-12-v2",
        "size_in_GB": 0.12,
        "sources": {
            "hf": "Xenova/ms-marco-MiniLM-L-12-v2",
        },
        "model_file": "onnx/model.onnx",
        "description": "MiniLM-L-12-v2 model optimized for re-ranking tasks.",
        "license": "apache-2.0",
    },
    {
        "model": "BAAI/bge-reranker-base",
        "size_in_GB": 1.04,
        "sources": {
            "hf": "BAAI/bge-reranker-base",
        },
        "model_file": "onnx/model.onnx",
        "description": "BGE reranker base model for cross-encoder re-ranking.",
        "license": "mit",
    },
    {
        "model": "jinaai/jina-reranker-v1-tiny-en",
        "size_in_GB": 0.13,
        "sources": {
            "hf": "jinaai/jina-reranker-v1-tiny-en",
        },
        "model_file": "onnx/model.onnx",
        "description": "Designed for blazing-fast re-ranking with 8K context length and fewer parameters than jina-reranker-v1-turbo-en.",
        "license": "apache-2.0",
    },
    {
        "model": "jinaai/jina-reranker-v1-turbo-en",
        "size_in_GB": 0.15,
        "sources": {
            "hf": "jinaai/jina-reranker-v1-turbo-en",
        },
        "model_file": "onnx/model.onnx",
        "description": "Designed for blazing-fast re-ranking with 8K context length.",
        "license": "apache-2.0",
    },
    {
        "model": "jinaai/jina-reranker-v2-base-multilingual",
        "size_in_GB": 1.11,
        "sources": {
            "hf": "jinaai/jina-reranker-v2-base-multilingual",
        },
        "model_file": "onnx/model.onnx",
        "description": "A multi-lingual reranker model for cross-encoder re-ranking with 1K context length and sliding window",
        "license": "cc-by-nc-4.0",
    },
]


class OnnxTextCrossEncoder(TextCrossEncoderBase, OnnxCrossEncoderModel):
    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        """Lists the supported models.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the model information.
        """
        return supported_onnx_models

    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence[OnnxProvider]] = None,
        cuda: bool = False,
        device_ids: Optional[List[int]] = None,
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
            device_ids (Optional[List[int]], optional): The list of device ids to use for data parallel processing in
                workers. Should be used with `cuda=True`, mutually exclusive with `providers`. Defaults to None.
            lazy_load (bool, optional): Whether to load the model during class initialization or on demand.
                Should be set to True when using multiple-gpu and parallel encoding. Defaults to False.
            device_id (Optional[int], optional): The device id to use for loading the model in the worker process.

        Raises:
            ValueError: If the model_name is not in the format <org>/<model> e.g. Xenova/ms-marco-MiniLM-L-6-v2.
        """
        super().__init__(model_name, cache_dir, threads, **kwargs)
        self.providers = providers
        self.lazy_load = lazy_load

        # List of device ids, that can be used for data parallel processing in workers
        self.device_ids = device_ids
        self.cuda = cuda

        if self.device_ids is not None and len(self.device_ids) > 1:
            logger.warning(
                "Parallel execution is currently not supported for cross encoders, "
                f"only the first device will be used for inference: {self.device_ids[0]}."
            )

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

    def load_onnx_model(self) -> None:
        self._load_onnx_model(
            model_dir=self._model_dir,
            model_file=self.model_description["model_file"],
            threads=self.threads,
            providers=self.providers,
            cuda=self.cuda,
            device_id=self.device_id,
        )

    def rerank(
        self,
        query: str,
        documents: Iterable[str],
        batch_size: int = 64,
        **kwargs,
    ) -> Iterable[float]:
        """Reranks documents based on their relevance to a given query.

        Args:
            query (str): The query string to which document relevance is calculated.
            documents (Iterable[str]): Iterable of documents to be reranked.
            batch_size (int, optional): The number of documents processed in each batch. Higher batch sizes improve speed
                                        but require more memory. Default is 64.
        Returns:
            Iterable[float]: An iterable of relevance scores for each document.
        """

        yield from self._rerank_documents(
            query=query, documents=documents, batch_size=batch_size, **kwargs
        )
