from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Type, Sequence

import numpy as np

from fastembed.common import OnnxProvider
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.utils import define_cache_dir
from fastembed.sparse.sparse_embedding_base import SparseEmbedding, SparseTextEmbeddingBase
from fastembed.text.onnx_text_model import OnnxTextModel, TextEmbeddingWorker

supported_bm42_models = [
    {
        "model": "Qdrant/bm42-all-minilm-l6-v2-attentions",
        "vocab_size": 30522,
        "description": "Light sparse embedding model, which assigns an importance score to each token in the text",
        "size_in_GB": 0.09,
        "sources": {
            "hf": "Qdrant/all_miniLM_L6_v2_with_attentions",
        },
        "model_file": "model.onnx",
    },
]


class Bm42(SparseTextEmbeddingBase, OnnxTextModel[SparseEmbedding]):

    def _post_process_onnx_output(
            self, output: OnnxOutputContext
    ) -> Iterable[SparseEmbedding]:

        token_ids = output.input_ids

        print(token_ids)

        for document in token_ids:
            for token_id in document:
                print(token_id, self.invert_vocab[token_id])

        raise NotImplementedError("TODO")

    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        """Lists the supported models.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the model information.
        """
        return supported_bm42_models

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

        self.invert_vocab = {}

        for token, idx in self.tokenizer.get_vocab().items():
            self.invert_vocab[idx] = token

        self.special_tokens = list(self.special_token_to_id.keys())
        self.special_tokens_ids = list(self.special_token_to_id.values())

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
    def _get_worker_class(cls) -> Type[TextEmbeddingWorker]:
        return Bm42EmbeddingWorker


class Bm42EmbeddingWorker(TextEmbeddingWorker):
    def init_embedding(
            self,
            model_name: str,
            cache_dir: str,
    ) -> Bm42:
        return Bm42(model_name=model_name, cache_dir=cache_dir, threads=1)

    def process(self, items: Iterable[Tuple[int, Any]]) -> Iterable[Tuple[int, Any]]:
        for idx, batch in items:
            onnx_output = self.model.onnx_embed(batch, ["attention_6"])
            yield idx, onnx_output
