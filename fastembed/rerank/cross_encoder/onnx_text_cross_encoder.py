from typing import List, Iterable, Dict, Any, Sequence, Optional

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
    },
    {
        "model": "Xenova/ms-marco-MiniLM-L-12-v2",
        "size_in_GB": 0.12,
        "sources": {
            "hf": "Xenova/ms-marco-MiniLM-L-12-v2",
        },
        "model_file": "onnx/model.onnx",
        "description": "MiniLM-L-12-v2 model optimized for re-ranking tasks.",
    },
    {
        "model": "BAAI/bge-reranker-base",
        "size_in_GB": 1.04,
        "sources": {
            "hf": "BAAI/bge-reranker-base",
        },
        "model_file": "onnx/model.onnx",
        "description": "BGE reranker base model for cross-encoder re-ranking.",
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
        **kwargs,
    ):
        """
        Args:
            model_name (str): The name of the model to use.
            cache_dir (str, optional): The path to the cache directory.
                                       Can be set using the `FASTEMBED_CACHE_PATH` env variable.
                                       Defaults to `fastembed_cache` in the system's temp directory.
            threads (int, optional): The number of threads single onnxruntime session can use. Defaults to None.
            providers (Optional[Sequence[OnnxProvider]]): The list of providers to use for the onnxruntime session.

        Raises:
            ValueError: If the model_name is not in the format <org>/<model> e.g. Xenova/ms-marco-MiniLM-L-6-v2.
        """
        super().__init__(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=threads,
            providers=providers,
            **kwargs,
        )

        model_description = self._get_model_description(model_name)
        self.cache_dir = define_cache_dir(cache_dir)
        self._model_dir = self.download_model(
            model_description, self.cache_dir, local_files_only=self._local_files_only
        )

        self.load_onnx_model(
            model_dir=self._model_dir,
            model_file=model_description["model_file"],
            threads=threads,
            providers=providers,
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
