from typing import Any, Dict, Iterable, List, Optional, Sequence, Type, Union

import numpy as np
from fastembed.common import OnnxProvider
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.utils import define_cache_dir
from fastembed.sparse.sparse_embedding_base import (
    SparseEmbedding,
    SparseTextEmbeddingBase,
)
from fastembed.text.onnx_text_model import OnnxTextModel, TextEmbeddingWorker

supported_splade_models = [
    {
        "model": "prithivida/Splade_PP_en_v1",
        "vocab_size": 30522,
        "description": "Independent Implementation of SPLADE++ Model for English.",
        "license": "apache-2.0",
        "size_in_GB": 0.532,
        "sources": {
            "hf": "Qdrant/SPLADE_PP_en_v1",
        },
        "model_file": "model.onnx",
    },
    {
        "model": "prithvida/Splade_PP_en_v1",
        "vocab_size": 30522,
        "description": "Independent Implementation of SPLADE++ Model for English.",
        "license": "apache-2.0",
        "size_in_GB": 0.532,
        "sources": {
            "hf": "Qdrant/SPLADE_PP_en_v1",
        },
        "model_file": "model.onnx",
    },
]


class SpladePP(SparseTextEmbeddingBase, OnnxTextModel[SparseEmbedding]):
    def _post_process_onnx_output(self, output: OnnxOutputContext) -> Iterable[SparseEmbedding]:
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
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        """Lists the supported models.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the model information.
        """
        return supported_splade_models

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

    def load_onnx_model(self) -> None:
        self._load_onnx_model(
            model_dir=self._model_dir,
            model_file=self.model_description["model_file"],
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
        **kwargs,
    ) -> Iterable[SparseEmbedding]:
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
    def _get_worker_class(cls) -> Type[TextEmbeddingWorker]:
        return SpladePPEmbeddingWorker


class SpladePPEmbeddingWorker(TextEmbeddingWorker):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs) -> SpladePP:
        return SpladePP(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
