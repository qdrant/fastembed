import math
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import mmh3
import numpy as np
from py_rust_stemmers import SnowballStemmer

from fastembed.common import OnnxProvider
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.utils import define_cache_dir
from fastembed.sparse.sparse_embedding_base import (
    SparseEmbedding,
    SparseTextEmbeddingBase,
)
from fastembed.text.onnx_text_model import OnnxTextModel, TextEmbeddingWorker

supported_bm42_models = [
    {
        "model": "Qdrant/bm42-all-minilm-l6-v2-attentions",
        "vocab_size": 30522,
        "description": "Light sparse embedding model, which assigns an importance score to each token in the text",
        "license": "apache-2.0",
        "size_in_GB": 0.09,
        "sources": {
            "hf": "Qdrant/all_miniLM_L6_v2_with_attentions",
        },
        "model_file": "model.onnx",
        "additional_files": ["stopwords.txt"],
        "requires_idf": True,
    },
]

MODEL_TO_LANGUAGE = {
    "Qdrant/bm42-all-minilm-l6-v2-attentions": "english",
}


class Bm42(SparseTextEmbeddingBase, OnnxTextModel[SparseEmbedding]):
    """
    Bm42 is an extension of BM25, which tries to better evaluate importance of tokens in the documents,
    by extracting attention weights from the transformer model.

    Traditional BM25 uses a count of tokens in the document to evaluate the importance of the token,
    but this approach doesn't work well with short documents or chunks of text, as almost all tokens
    there are unique.

    BM42 addresses this issue by replacing the token count with the attention weights from the transformer model.
    This allows sparse embeddings to work well with short documents, handle rare tokens and leverage traditional NLP
    techniques like stemming and stopwords.

    WARNING: This model is expected to be used with `modifier="idf"` in the sparse vector index of Qdrant.
    """

    ONNX_OUTPUT_NAMES = ["attention_6"]

    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence[OnnxProvider]] = None,
        alpha: float = 0.5,
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
            providers (Optional[Sequence[OnnxProvider]], optional): The providers to use for onnxruntime.
            alpha (float, optional): Parameter, that defines the importance of the token weight in the document
                versus the importance of the token frequency in the corpus. Defaults to 0.5, based on empirical testing.
                It is recommended to only change this parameter based on training data for a specific dataset.
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

        self.invert_vocab = {}

        self.special_tokens = set()
        self.special_tokens_ids = set()
        self.punctuation = set(string.punctuation)
        self.stopwords = set(self._load_stopwords(self._model_dir))
        self.stemmer = SnowballStemmer(MODEL_TO_LANGUAGE[model_name])
        self.alpha = alpha

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
        for token, idx in self.tokenizer.get_vocab().items():
            self.invert_vocab[idx] = token
        self.special_tokens = set(self.special_token_to_id.keys())
        self.special_tokens_ids = set(self.special_token_to_id.values())
        self.stopwords = set(self._load_stopwords(self._model_dir))

    def _filter_pair_tokens(self, tokens: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
        result = []
        for token, value in tokens:
            if token in self.stopwords or token in self.punctuation:
                continue
            result.append((token, value))
        return result

    def _stem_pair_tokens(self, tokens: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
        result = []
        for token, value in tokens:
            processed_token = self.stemmer.stem_word(token)
            result.append((processed_token, value))
        return result

    @classmethod
    def _aggregate_weights(
        cls, tokens: List[Tuple[str, List[int]]], weights: List[float]
    ) -> List[Tuple[str, float]]:
        result = []
        for token, idxs in tokens:
            sum_weight = sum(weights[idx] for idx in idxs)
            result.append((token, sum_weight))
        return result

    def _reconstruct_bpe(
        self, bpe_tokens: Iterable[Tuple[int, str]]
    ) -> List[Tuple[str, List[int]]]:
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

    def _rescore_vector(self, vector: Dict[str, float]) -> Dict[int, float]:
        """
        Orders all tokens in the vector by their importance and generates a new score based on the importance order.
        So that the scoring doesn't depend on absolute values assigned by the model, but on the relative importance.
        """

        new_vector = {}

        for token, value in vector.items():
            token_id = abs(mmh3.hash(token))
            # Examples:
            # Num 0: Log(1/1 + 1) = 0.6931471805599453
            # Num 1: Log(1/2 + 1) = 0.4054651081081644
            # Num 2: Log(1/3 + 1) = 0.28768207245178085
            new_vector[token_id] = math.log(1.0 + value) ** self.alpha  # value

        return new_vector

    def _post_process_onnx_output(self, output: OnnxOutputContext) -> Iterable[SparseEmbedding]:
        if output.input_ids is None:
            raise ValueError("input_ids must be provided for document post-processing")

        token_ids_batch = output.input_ids

        # attention_value shape: (batch_size, num_heads, num_tokens, num_tokens)
        pooled_attention = np.mean(output.model_output[:, :, 0], axis=1) * output.attention_mask

        for document_token_ids, attention_value in zip(token_ids_batch, pooled_attention):
            document_tokens_with_ids = (
                (idx, self.invert_vocab[token_id])
                for idx, token_id in enumerate(document_token_ids)
            )

            reconstructed = self._reconstruct_bpe(document_tokens_with_ids)

            filtered = self._filter_pair_tokens(reconstructed)

            stemmed = self._stem_pair_tokens(filtered)

            weighted = self._aggregate_weights(stemmed, attention_value)

            max_token_weight = {}

            for token, weight in weighted:
                max_token_weight[token] = max(max_token_weight.get(token, 0), weight)

            rescored = self._rescore_vector(max_token_weight)

            yield SparseEmbedding.from_dict(rescored)

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
            alpha=self.alpha,
        )

    @classmethod
    def _query_rehash(cls, tokens: Iterable[str]) -> Dict[int, float]:
        result = {}
        for token in tokens:
            token_id = abs(mmh3.hash(token))
            result[token_id] = 1.0
        return result

    def query_embed(self, query: Union[str, Iterable[str]], **kwargs) -> Iterable[SparseEmbedding]:
        """
        To emulate BM25 behaviour, we don't need to use smart weights in the query, and
        it's enough to just hash the tokens and assign a weight of 1.0 to them.
        It is also faster, as we don't need to run the model for the query.
        """
        if isinstance(query, str):
            query = [query]

        if not hasattr(self, "model") or self.model is None:
            self.load_onnx_model()

        for text in query:
            encoded = self.tokenizer.encode(text)
            document_tokens_with_ids = enumerate(encoded.tokens)
            reconstructed = self._reconstruct_bpe(document_tokens_with_ids)
            filtered = self._filter_pair_tokens(reconstructed)
            stemmed = self._stem_pair_tokens(filtered)

            yield SparseEmbedding.from_dict(self._query_rehash(token for token, _ in stemmed))

    @classmethod
    def _get_worker_class(cls) -> Type[TextEmbeddingWorker]:
        return Bm42TextEmbeddingWorker


class Bm42TextEmbeddingWorker(TextEmbeddingWorker):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs) -> Bm42:
        return Bm42(
            model_name=model_name,
            cache_dir=cache_dir,
            **kwargs,
        )
