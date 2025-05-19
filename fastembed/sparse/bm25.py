import os
from collections import defaultdict
from multiprocessing import get_all_start_methods
from pathlib import Path
from typing import Any, Iterable, Optional, Type, Union

import mmh3
import numpy as np
from py_rust_stemmers import SnowballStemmer
from fastembed.common.utils import (
    define_cache_dir,
    iter_batch,
    get_all_punctuation,
    remove_non_alphanumeric,
)
from fastembed.parallel_processor import ParallelWorkerPool, Worker
from fastembed.sparse.sparse_embedding_base import (
    SparseEmbedding,
    SparseTextEmbeddingBase,
)
from fastembed.sparse.utils.tokenizer import SimpleTokenizer
from fastembed.common.model_description import SparseModelDescription, ModelSource


supported_languages = [
    "arabic",
    "danish",
    "dutch",
    "english",
    "finnish",
    "french",
    "german",
    "greek",
    "hungarian",
    "italian",
    "norwegian",
    "portuguese",
    "romanian",
    "russian",
    "spanish",
    "swedish",
    "tamil",
    "turkish",
]

supported_bm25_models: list[SparseModelDescription] = [
    SparseModelDescription(
        model="Qdrant/bm25",
        vocab_size=0,
        description="BM25 as sparse embeddings meant to be used with Qdrant",
        license="apache-2.0",
        size_in_GB=0.01,
        sources=ModelSource(hf="Qdrant/bm25"),
        additional_files=[f"{lang}.txt" for lang in supported_languages],
        requires_idf=True,
        model_file="mock.file",
    ),
]


class Bm25(SparseTextEmbeddingBase):
    """Implements traditional BM25 in a form of sparse embeddings.
    Uses a count of tokens in the document to evaluate the importance of the token.

    WARNING: This model is expected to be used with `modifier="idf"` in the sparse vector index of Qdrant.

    BM25 formula:

    score(q, d) = SUM[ IDF(q_i) * (f(q_i, d) * (k + 1)) / (f(q_i, d) + k * (1 - b + b * (|d| / avg_len))) ],

    where IDF is the inverse document frequency, computed on Qdrant's side
    f(q_i, d) is the term frequency of the token q_i in the document d
    k, b, avg_len are hyperparameters, described below.

    Args:
        model_name (str): The name of the model to use.
        cache_dir (str, optional): The path to the cache directory.
            Can be set using the `FASTEMBED_CACHE_PATH` env variable.
            Defaults to `fastembed_cache` in the system's temp directory.
        k (float, optional): The k parameter in the BM25 formula. Defines the saturation of the term frequency.
            I.e. defines how fast the moment when additional terms stop to increase the score. Defaults to 1.2.
        b (float, optional): The b parameter in the BM25 formula. Defines the importance of the document length.
            Defaults to 0.75.
        avg_len (float, optional): The average length of the documents in the corpus. Defaults to 256.0.
        language (str): Specifies the language for the stemmer.
        disable_stemmer (bool): Disable the stemmer.
    Raises:
        ValueError: If the model_name is not in the format <org>/<model> e.g. BAAI/bge-base-en.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        k: float = 1.2,
        b: float = 0.75,
        avg_len: float = 256.0,
        language: str = "english",
        token_max_length: int = 40,
        disable_stemmer: bool = False,
        specific_model_path: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(model_name, cache_dir, **kwargs)

        if language not in supported_languages:
            raise ValueError(f"{language} language is not supported")
        else:
            self.language = language

        self.k = k
        self.b = b
        self.avg_len = avg_len

        model_description = self._get_model_description(model_name)
        self.cache_dir = str(define_cache_dir(cache_dir))

        self._specific_model_path = specific_model_path
        self._model_dir = self.download_model(
            model_description,
            self.cache_dir,
            local_files_only=self._local_files_only,
            specific_model_path=self._specific_model_path,
        )

        self.token_max_length = token_max_length
        self.punctuation = set(get_all_punctuation())
        self.disable_stemmer = disable_stemmer

        if disable_stemmer:
            self.stopwords: set[str] = set()
            self.stemmer = None
        else:
            self.stopwords = set(self._load_stopwords(self._model_dir, self.language))
            self.stemmer = SnowballStemmer(language)

        self.tokenizer = SimpleTokenizer

    @classmethod
    def _list_supported_models(cls) -> list[SparseModelDescription]:
        """Lists the supported models.

        Returns:
            list[SparseModelDescription]: A list of SparseModelDescription objects containing the model information.
        """
        return supported_bm25_models

    @classmethod
    def _load_stopwords(cls, model_dir: Path, language: str) -> list[str]:
        stopwords_path = model_dir / f"{language}.txt"
        if not stopwords_path.exists():
            return []

        with open(stopwords_path, "r") as f:
            return f.read().splitlines()

    def _embed_documents(
        self,
        model_name: str,
        cache_dir: str,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        local_files_only: bool = False,
        specific_model_path: Optional[str] = None,
    ) -> Iterable[SparseEmbedding]:
        is_small = False

        if isinstance(documents, str):
            documents = [documents]
            is_small = True

        if isinstance(documents, list):
            if len(documents) < batch_size:
                is_small = True

        if parallel is None or is_small:
            for batch in iter_batch(documents, batch_size):
                yield from self.raw_embed(batch)
        else:
            if parallel == 0:
                parallel = os.cpu_count()

            start_method = "forkserver" if "forkserver" in get_all_start_methods() else "spawn"
            params = {
                "model_name": model_name,
                "cache_dir": cache_dir,
                "k": self.k,
                "b": self.b,
                "avg_len": self.avg_len,
                "language": self.language,
                "token_max_length": self.token_max_length,
                "disable_stemmer": self.disable_stemmer,
                "local_files_only": local_files_only,
                "specific_model_path": specific_model_path,
            }
            pool = ParallelWorkerPool(
                num_workers=parallel or 1,
                worker=self._get_worker_class(),
                start_method=start_method,
            )
            for batch in pool.ordered_map(iter_batch(documents, batch_size), **params):
                for record in batch:
                    yield record  # type: ignore

    def embed(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs: Any,
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
            local_files_only=self._local_files_only,
            specific_model_path=self._specific_model_path,
        )

    def _stem(self, tokens: list[str]) -> list[str]:
        stemmed_tokens: list[str] = []
        for token in tokens:
            lower_token = token.lower()

            if token in self.punctuation:
                continue

            if lower_token in self.stopwords:
                continue

            if len(token) > self.token_max_length:
                continue

            stemmed_token = self.stemmer.stem_word(lower_token) if self.stemmer else lower_token

            if stemmed_token:
                stemmed_tokens.append(stemmed_token)
        return stemmed_tokens

    def raw_embed(
        self,
        documents: list[str],
    ) -> list[SparseEmbedding]:
        embeddings: list[SparseEmbedding] = []
        for document in documents:
            document = remove_non_alphanumeric(document)
            tokens = self.tokenizer.tokenize(document)
            stemmed_tokens = self._stem(tokens)
            token_id2value = self._term_frequency(stemmed_tokens)
            embeddings.append(SparseEmbedding.from_dict(token_id2value))
        return embeddings

    def _term_frequency(self, tokens: list[str]) -> dict[int, float]:
        """Calculate the term frequency part of the BM25 formula.

        (
            f(q_i, d) * (k + 1)
        ) / (
            f(q_i, d) + k * (1 - b + b * (|d| / avg_len))
        )

        Args:
            tokens (list[str]): The list of tokens in the document.

        Returns:
            dict[int, float]: The token_id to term frequency mapping.
        """
        tf_map: dict[int, float] = {}
        counter: defaultdict[str, int] = defaultdict(int)
        for stemmed_token in tokens:
            counter[stemmed_token] += 1

        doc_len = len(tokens)
        for stemmed_token in counter:
            token_id = self.compute_token_id(stemmed_token)
            num_occurrences = counter[stemmed_token]
            tf_map[token_id] = num_occurrences * (self.k + 1)
            tf_map[token_id] /= num_occurrences + self.k * (
                1 - self.b + self.b * doc_len / self.avg_len
            )
        return tf_map

    @classmethod
    def compute_token_id(cls, token: str) -> int:
        return abs(mmh3.hash(token))

    def query_embed(
        self, query: Union[str, Iterable[str]], **kwargs: Any
    ) -> Iterable[SparseEmbedding]:
        """To emulate BM25 behaviour, we don't need to use weights in the query, and
        it's enough to just hash the tokens and assign a weight of 1.0 to them.
        """
        if isinstance(query, str):
            query = [query]

        for text in query:
            text = remove_non_alphanumeric(text)
            tokens = self.tokenizer.tokenize(text)
            stemmed_tokens = self._stem(tokens)
            token_ids = np.array(
                list(set(self.compute_token_id(token) for token in stemmed_tokens)),
                dtype=np.int32,
            )
            values = np.ones_like(token_ids)
            yield SparseEmbedding(indices=token_ids, values=values)

    @classmethod
    def _get_worker_class(cls) -> Type["Bm25Worker"]:
        return Bm25Worker


class Bm25Worker(Worker):
    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs: Any,
    ):
        self.model = self.init_embedding(model_name, cache_dir, **kwargs)

    @classmethod
    def start(cls, model_name: str, cache_dir: str, **kwargs: Any) -> "Bm25Worker":
        return cls(model_name=model_name, cache_dir=cache_dir, **kwargs)

    def process(
        self, items: Iterable[tuple[int, Any]]
    ) -> Iterable[tuple[int, list[SparseEmbedding]]]:
        for idx, batch in items:
            onnx_output = self.model.raw_embed(batch)
            yield idx, onnx_output

    @staticmethod
    def init_embedding(model_name: str, cache_dir: str, **kwargs: Any) -> Bm25:
        return Bm25(model_name=model_name, cache_dir=cache_dir, **kwargs)
