import string
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Type, Sequence

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
        "additional_files": [
            "stopwords.txt"
        ],
    },
]

MODEL_TO_LANGUAGE = {
    "Qdrant/bm42-all-minilm-l6-v2-attentions": "english",
}


class Bm42(SparseTextEmbeddingBase, OnnxTextModel[SparseEmbedding]):

    def _filter_pair_tokens(self, tokens: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
        result = []
        for token, value in tokens:
            if token in self.stopwords or token in self.punctuation:
                continue
            result.append((token, value))
        return result

    def _reconstruct_bpe(self, bpe_tokens: Iterable[Tuple[int, str]]) -> List[Tuple[str, List[int]]]:

        result = []
        acc = ""
        acc_idx = []

        continuing_subword_prefix = self.tokenizer.model.continuing_subword_prefix
        continuing_subword_prefix_len = len(continuing_subword_prefix)

        for idx, token in bpe_tokens:
            if token in self.special_tokens:
                continue

            if token.startswith(continuing_subword_prefix):
                acc += token[continuing_subword_prefix_len:]
                acc_idx.append(idx)
            else:
                if acc:
                    result.append((acc, acc_idx))
                    acc_idx = []
                acc = token
                acc_idx.append(idx)

        if acc:
            result.append((acc, acc_idx))

        return result

    def _post_process_onnx_output(
            self, output: OnnxOutputContext
    ) -> Iterable[SparseEmbedding]:

        token_ids_batch = output.input_ids

        for document_token_ids in token_ids_batch:
            document_tokens_with_ids = ((token_id, self.invert_vocab[token_id]) for token_id in document_token_ids)

            reconstructed = self._reconstruct_bpe(document_tokens_with_ids)

            filtered = self._filter_pair_tokens(reconstructed)

            for x in reconstructed:
                print(x)

        raise NotImplementedError("TODO")

    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        """Lists the supported models.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the model information.
        """
        return supported_bm42_models

    @classmethod
    def _load_stopwords(cls, model_dir: Path) -> List[str]:
        stopwords_path = model_dir / "stopwords.txt"
        if not stopwords_path.exists():
            return []

        with open(stopwords_path, "r") as f:
            return f.read().splitlines()

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

        self.special_tokens = set(self.special_token_to_id.keys())
        self.special_tokens_ids = set(self.special_token_to_id.values())
        self.punctuation = set(string.punctuation)
        self.stopwords = set(self._load_stopwords(model_dir))

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
