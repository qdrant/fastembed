from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Type, Sequence

import numpy as np

from fastembed.common.onnx_model import EmbeddingWorker, OnnxModel, OnnxProvider
from fastembed.common.utils import define_cache_dir
from fastembed.sparse.sparse_embedding_base import SparseEmbedding, SparseTextEmbeddingBase

supported_splade_models = [
    {
        "model": "prithvida/Splade_PP_en_v1",
        "vocab_size": 30522,
        "description": "Misspelled version of the model. Retained for backward compatibility. Independent Implementation of SPLADE++ Model for English",
        "size_in_GB": 0.532,
        "sources": {
            "hf": "Qdrant/SPLADE_PP_en_v1",
        },
        "model_file": "model.onnx",
    },
    {
        "model": "prithivida/Splade_PP_en_v1",
        "vocab_size": 30522,
        "description": "Independent Implementation of SPLADE++ Model for English",
        "size_in_GB": 0.532,
        "sources": {
            "hf": "Qdrant/SPLADE_PP_en_v1",
        },
        "model_file": "model.onnx",
    },
]


class SpladePP(SparseTextEmbeddingBase, OnnxModel[SparseEmbedding]):
    @classmethod
    def _post_process_onnx_output(
        cls, output: Tuple[np.ndarray, np.ndarray]
    ) -> Iterable[SparseEmbedding]:
        logits, attention_mask = output
        relu_log = np.log(1 + np.maximum(logits, 0))

        weighted_log = relu_log * np.expand_dims(attention_mask, axis=-1)

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

        model_description = self._get_model_description(model_name)
        cache_dir = define_cache_dir(cache_dir)

        model_dir = self.download_model(
            model_description, cache_dir, local_files_only=self._local_files_only
        )

        self.load_onnx_model(
            model_dir=model_dir,
            model_file=model_description["model_file"],
            threads=threads,
            providers=providers,
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
        )

    @classmethod
    def _get_worker_class(cls) -> Type[EmbeddingWorker]:
        return SpladePPEmbeddingWorker


class SpladePPEmbeddingWorker(EmbeddingWorker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
    ) -> SpladePP:
        return SpladePP(model_name=model_name, cache_dir=cache_dir, threads=1)
