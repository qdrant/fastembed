from typing import Any, Iterable, Optional, Sequence, Type

from loguru import logger

from fastembed.common import OnnxProvider
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.utils import define_cache_dir
from fastembed.rerank.cross_encoder.onnx_text_model import (
    OnnxCrossEncoderModel,
    TextRerankerWorker,
)
from fastembed.rerank.cross_encoder.text_cross_encoder_base import TextCrossEncoderBase
from fastembed.common.model_description import BaseModelDescription, ModelSource

supported_onnx_models: list[BaseModelDescription] = [
    BaseModelDescription(
        model="Xenova/ms-marco-MiniLM-L-6-v2",
        description="MiniLM-L-6-v2 model optimized for re-ranking tasks.",
        license="apache-2.0",
        size_in_GB=0.08,
        sources=ModelSource(hf="Xenova/ms-marco-MiniLM-L-6-v2"),
        model_file="onnx/model.onnx",
    ),
    BaseModelDescription(
        model="Xenova/ms-marco-MiniLM-L-12-v2",
        description="MiniLM-L-12-v2 model optimized for re-ranking tasks.",
        license="apache-2.0",
        size_in_GB=0.12,
        sources=ModelSource(hf="Xenova/ms-marco-MiniLM-L-12-v2"),
        model_file="onnx/model.onnx",
    ),
    BaseModelDescription(
        model="BAAI/bge-reranker-base",
        description="BGE reranker base model for cross-encoder re-ranking.",
        license="mit",
        size_in_GB=1.04,
        sources=ModelSource(hf="BAAI/bge-reranker-base"),
        model_file="onnx/model.onnx",
    ),
    BaseModelDescription(
        model="jinaai/jina-reranker-v1-tiny-en",
        description="Designed for blazing-fast re-ranking with 8K context length and fewer parameters than jina-reranker-v1-turbo-en.",
        license="apache-2.0",
        size_in_GB=0.13,
        sources=ModelSource(hf="jinaai/jina-reranker-v1-tiny-en"),
        model_file="onnx/model.onnx",
    ),
    BaseModelDescription(
        model="jinaai/jina-reranker-v1-turbo-en",
        description="Designed for blazing-fast re-ranking with 8K context length.",
        license="apache-2.0",
        size_in_GB=0.15,
        sources=ModelSource(hf="jinaai/jina-reranker-v1-turbo-en"),
        model_file="onnx/model.onnx",
    ),
    BaseModelDescription(
        model="jinaai/jina-reranker-v2-base-multilingual",
        description="A multi-lingual reranker model for cross-encoder re-ranking with 1K context length and sliding window",
        license="cc-by-nc-4.0",
        size_in_GB=1.11,
        sources=ModelSource(hf="jinaai/jina-reranker-v2-base-multilingual"),
        model_file="onnx/model.onnx",
    ),
]


class OnnxTextCrossEncoder(TextCrossEncoderBase, OnnxCrossEncoderModel):
    @classmethod
    def _list_supported_models(cls) -> list[BaseModelDescription]:
        """Lists the supported models.

        Returns:
            list[BaseModelDescription]: A list of BaseModelDescription objects containing the model information.
        """
        return supported_onnx_models

    def __init__(
        self,
        model_name: str,
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
        Initializes an ONNX-based cross-encoder model for text re-ranking.
        
        Configures model selection, caching, threading, device assignment, ONNX runtime providers, and model loading behavior. Downloads and prepares the ONNX model for inference, with support for custom model paths and lazy loading. Raises a ValueError if the model name format is invalid.
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

    def load_onnx_model(self) -> None:
        self._load_onnx_model(
            model_dir=self._model_dir,
            model_file=self.model_description.model_file,
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
        **kwargs: Any,
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

    def rerank_pairs(
        self,
        pairs: Iterable[tuple[str, str]],
        batch_size: int = 64,
        parallel: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterable[float]:
        """
        Reranks pairs of texts using the ONNX cross-encoder model.
        
        Args:
            pairs: An iterable of (query, document) string tuples to be scored.
            batch_size: Number of pairs to process in each batch. Defaults to 64.
            parallel: Optional number of parallel workers for processing.
        
        Yields:
            Relevance scores as floats for each input pair, in order.
        """
        yield from self._rerank_pairs(
            model_name=self.model_name,
            cache_dir=str(self.cache_dir),
            pairs=pairs,
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
    def _get_worker_class(cls) -> Type[TextRerankerWorker]:
        return TextCrossEncoderWorker

    def _post_process_onnx_output(
        self, output: OnnxOutputContext, **kwargs: Any
    ) -> Iterable[float]:
        return (float(elem) for elem in output.model_output)


class TextCrossEncoderWorker(TextRerankerWorker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs: Any,
    ) -> OnnxTextCrossEncoder:
        return OnnxTextCrossEncoder(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
