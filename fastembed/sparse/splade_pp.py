from typing import Any, Iterable, Optional, Sequence, Type, Union

import numpy as np
from fastembed.common import OnnxProvider
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.utils import define_cache_dir
from fastembed.sparse.sparse_embedding_base import (
    SparseEmbedding,
    SparseTextEmbeddingBase,
)
from fastembed.text.onnx_text_model import OnnxTextModel, TextEmbeddingWorker
from fastembed.common.model_description import SparseModelDescription, ModelSource

supported_splade_models: list[SparseModelDescription] = [
    SparseModelDescription(
        model="prithivida/Splade_PP_en_v1",
        vocab_size=30522,
        description="Independent Implementation of SPLADE++ Model for English.",
        license="apache-2.0",
        size_in_GB=0.532,
        sources=ModelSource(hf="Qdrant/Splade_PP_en_v1"),
        model_file="model.onnx",
    ),
    SparseModelDescription(
        model="prithvida/Splade_PP_en_v1",
        vocab_size=30522,
        description="Independent Implementation of SPLADE++ Model for English.",
        license="apache-2.0",
        size_in_GB=0.532,
        sources=ModelSource(hf="Qdrant/Splade_PP_en_v1"),
        model_file="model.onnx",
    ),
]


class SpladePP(SparseTextEmbeddingBase, OnnxTextModel[SparseEmbedding]):
    def _post_process_onnx_output(
        self, output: OnnxOutputContext, **kwargs: Any
    ) -> Iterable[SparseEmbedding]:
        if output.attention_mask is None:
            raise ValueError("attention_mask must be provided for document post-processing")

        relu_log = np.log(1 + np.maximum(output.model_output, 0))

        weighted_log = relu_log * np.expand_dims(output.attention_mask, axis=-1)

        scores = np.max(weighted_log, axis=1)

        # Score matrix of shape (batch_size, vocab_size)
        # Most of the values are 0, only a few are non-zero
        for row_scores in scores:
            indices = row_scores.nonzero()[0]
            scores = row_scores[indices]
            yield SparseEmbedding(values=scores, indices=indices)

    @classmethod
    def _list_supported_models(cls) -> list[SparseModelDescription]:
        """Lists the supported models.

        Returns:
            list[SparseModelDescription]: A list of SparseModelDescription objects containing the model information.
        """
        return supported_splade_models

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
        Initializes a SPLADE++ sparse embedding model instance with specified configuration.
        
        Configures model loading, device selection, threading, and ONNX runtime options. Downloads the model files if necessary and loads the ONNX model immediately unless lazy loading is enabled.
        
        Raises:
            ValueError: If the model_name is not in the format <org>/<model> (e.g., BAAI/bge-base-en).
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

    def load_onnx_model(self) -> None:
        self._load_onnx_model(
            model_dir=self._model_dir,
            model_file=self.model_description.model_file,
            threads=self.threads,
            providers=self.providers,
            cuda=self.cuda,
            device_id=self.device_id,
        )

    def embed(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterable[SparseEmbedding]:
        """
        Encodes one or more documents into sparse embeddings.
        
        Args:
            documents: A single document or an iterable of documents to embed.
            batch_size: Number of documents to process per batch.
            parallel: Number of parallel workers to use for encoding. If >1, enables data-parallel processing; if 0, uses all available cores; if None, uses default threading.
        
        Returns:
            An iterable of SparseEmbedding objects, one for each input document.
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
    def _get_worker_class(cls) -> Type[TextEmbeddingWorker[SparseEmbedding]]:
        return SpladePPEmbeddingWorker


class SpladePPEmbeddingWorker(TextEmbeddingWorker[SparseEmbedding]):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs: Any) -> SpladePP:
        return SpladePP(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
