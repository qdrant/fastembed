import itertools
import json
import os
import warnings
from multiprocessing import get_all_start_methods
from pathlib import Path
from typing import Any, Iterable, Type

import numpy as np
from numpy.typing import NDArray
from tokenizers import Tokenizer

from fastembed.common.model_description import DenseModelDescription, ModelSource
from fastembed.common.safetensors_utils import load_safetensors
from fastembed.common.types import Device, NumpyArray
from fastembed.common.utils import define_cache_dir, iter_batch, normalize
from fastembed.parallel_processor import ParallelWorkerPool, Worker
from fastembed.text.text_embedding_base import TextEmbeddingBase

# Same cap as the onnx exports of these models and model2vec's default `max_length`.
MAX_LENGTH = 512

supported_static_models: list[DenseModelDescription] = [
    DenseModelDescription(
        model="minishlab/potion-base-8M",
        dim=256,
        description=(
            "Text embeddings, Unimodal (text), English, 512 input tokens truncation, "
            "Prefixes for queries/documents: not necessary, 2024 year."
        ),
        license="mit",
        size_in_GB=0.030,
        sources=ModelSource(hf="minishlab/potion-base-8M"),
        model_file="model.safetensors",
    ),
    DenseModelDescription(
        model="minishlab/potion-retrieval-32M",
        dim=512,
        description=(
            "Text embeddings, Unimodal (text), English, 512 input tokens truncation, "
            "Prefixes for queries/documents: not necessary, 2025 year."
        ),
        license="mit",
        size_in_GB=0.129,
        sources=ModelSource(hf="minishlab/potion-retrieval-32M"),
        model_file="model.safetensors",
    ),
    DenseModelDescription(
        model="minishlab/potion-multilingual-128M",
        dim=256,
        description=(
            "Text embeddings, Unimodal (text), Multilingual, 512 input tokens truncation, "
            "Prefixes for queries/documents: not necessary, 2025 year."
        ),
        license="mit",
        size_in_GB=0.512,
        sources=ModelSource(hf="minishlab/potion-multilingual-128M"),
        model_file="model.safetensors",
    ),
]


def load_static_tokenizer(model_dir: Path) -> tuple[Tokenizer, int | None]:
    """Load the tokenizer of a static model: truncated to MAX_LENGTH, without padding.

    Returns the tokenizer and the id of its unknown token, if it has one. Unknown tokens carry no
    meaning in a static model, so they are left out of the mean, as model2vec does.
    """
    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise ValueError(f"Could not find tokenizer.json in {model_dir}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=MAX_LENGTH)
    tokenizer.no_padding()

    with open(tokenizer_path, encoding="utf-8") as f:
        tokenizer_model = json.load(f)["model"]
    if tokenizer_model.get("type") == "Unigram":
        unk_token_id = tokenizer_model.get("unk_id")
    else:
        unk_token = tokenizer_model.get("unk_token")
        vocab = tokenizer_model.get("vocab")
        unk_token_id = vocab.get(unk_token) if unk_token and isinstance(vocab, dict) else None
    return tokenizer, unk_token_id


class StaticTextEmbedding(TextEmbeddingBase):
    """Static (model2vec) embeddings: the mean of the token vectors, computed with numpy.

    There is no network to run, so tokenization dominates the cost. The vectors are read straight
    from `model.safetensors` and pooled without the padding an onnx graph needs.
    """

    # Cells of the (documents x distinct tokens) count matrix pooled per matrix product. It bounds
    # the temporary memory of pooling to about 60 MiB, however long, diverse or many the documents
    # are.
    POOLING_CELLS = 4_000_000

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        return supported_static_models

    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        threads: int | None = None,
        lazy_load: bool = False,
        specific_model_path: str | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            model_name (str): The name of the model to use.
            cache_dir (str, optional): The path to the cache directory.
                Can be set using the `FASTEMBED_CACHE_PATH` env variable.
                Defaults to `fastembed_cache` in the system's temp directory.
            threads (int, optional): Not supported, a warning is raised if set. Tokenization runs on
                the thread pool of the tokenizers library, which `TOKENIZERS_PARALLELISM=false`
                turns off.
            lazy_load (bool, optional): Whether to load the model during class initialization or
                on demand. Defaults to False.
            specific_model_path (str, optional): A local directory with `model.safetensors` and
                `tokenizer.json` to load instead of downloading the model.
        """
        # TextEmbedding passes its onnx runtime options to every model class. Static models run
        # on the CPU with numpy and ignore them, so only options set to something else than their
        # default are reported.
        ignored = {"threads": threads} if threads is not None else {}
        for option, default in (
            ("providers", None),
            ("cuda", Device.AUTO),
            ("device_ids", None),
            ("enable_cpu_mem_arena", None),
        ):
            value = kwargs.pop(option, default)
            if value != default:
                ignored[option] = value
        kwargs.pop("device_id", None)  # set by parallel workers, not by users
        if ignored:
            warnings.warn(
                f"{model_name} is a static model computed with numpy on the CPU, "
                f"so these options have no effect: {', '.join(sorted(ignored))}",
                UserWarning,
                stacklevel=3,
            )
        super().__init__(model_name, cache_dir, threads, **kwargs)
        self.lazy_load = lazy_load
        self.model_description = self._get_model_description(model_name)
        self.cache_dir = str(define_cache_dir(cache_dir))
        self._specific_model_path = specific_model_path
        self._model_dir = self.download_model(
            self.model_description,
            self.cache_dir,
            local_files_only=self._local_files_only,
            specific_model_path=self._specific_model_path,
        )

        self.embeddings: NumpyArray | None = None
        self.weights: NumpyArray | None = None
        self.token_mapping: NumpyArray | None = None
        self.tokenizer: Tokenizer | None = None
        self.unk_token_id: int | None = None
        if not self.lazy_load:
            self.load_model()

    def load_model(self) -> None:
        tensors = load_safetensors(self._model_dir / self.model_description.model_file)
        if "embeddings" not in tensors:
            raise ValueError(f"No `embeddings` tensor in {self._model_dir}")
        self.embeddings = tensors["embeddings"]
        # Vocabulary-quantized model2vec models map token ids to shared rows and reweight them.
        self.weights = tensors.get("weights")
        self.token_mapping = tensors.get("mapping")
        self.tokenizer, self.unk_token_id = load_static_tokenizer(self._model_dir)

    def _ensure_model(self) -> None:
        if self.embeddings is None or self.tokenizer is None:
            self.load_model()

    def _token_ids(self, documents: list[str]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Tokenize a batch into one flat id array plus the number of ids of each document."""
        assert self.tokenizer is not None
        # encode_batch_fast skips the character offsets, which are not needed here.
        encodings = self.tokenizer.encode_batch_fast(documents, add_special_tokens=False)
        id_lists = [encoding.ids for encoding in encodings]
        lengths = np.fromiter(map(len, id_lists), dtype=np.int64, count=len(documents))
        ids = np.fromiter(
            itertools.chain.from_iterable(id_lists), dtype=np.int64, count=int(lengths.sum())
        )
        return ids, lengths

    def raw_embed(self, documents: list[str]) -> NumpyArray:
        """Embed one batch of documents, without batching or parallelism."""
        self._ensure_model()
        assert self.embeddings is not None
        ids, lengths = self._token_ids(documents)

        if self.unk_token_id is not None:
            known = ids != self.unk_token_id
            if not known.all():
                doc_index = np.repeat(np.arange(len(documents)), lengths)
                lengths = np.bincount(doc_index[known], minlength=len(documents))
                ids = ids[known]

        sums = self._sum_rows(ids, lengths)
        # The mean and the sum point the same way, so normalizing the sum gives the mean's
        # direction. Empty documents stay zero vectors.
        return normalize(sums)

    def _sum_rows(self, ids: NDArray[np.int64], lengths: NDArray[np.int64]) -> NumpyArray:
        """Sum the vectors of each document's tokens.

        The sums are a product of a (documents x distinct tokens) count matrix with the rows of
        those tokens: one matrix multiplication, several times faster than segment sums with
        np.add.reduceat or a loop over the documents. Documents whose count matrix would exceed
        POOLING_CELLS are halved until it fits, which bounds the temporary memory. A single
        document sums its rows directly.
        """
        assert self.embeddings is not None
        if not ids.size:
            return np.zeros((len(lengths), self.embeddings.shape[1]), dtype=np.float32)
        rows = ids if self.token_mapping is None else self.token_mapping[ids]
        if len(lengths) == 1:
            # A single text, typically a query: summing its rows skips the fixed cost of building
            # a count matrix, which dominates for short texts.
            vectors = self.embeddings[rows]
            if self.weights is not None:
                vectors = vectors * self.weights[ids, None]
            return vectors.sum(axis=0, keepdims=True, dtype=np.float32)
        distinct_rows, row_index = np.unique(rows, return_inverse=True)
        if len(lengths) > 1 and len(lengths) * len(distinct_rows) > self.POOLING_CELLS:
            half = len(lengths) // 2
            split = int(lengths[:half].sum())
            return np.concatenate(
                [
                    self._sum_rows(ids[:split], lengths[:half]),
                    self._sum_rows(ids[split:], lengths[half:]),
                ]
            )
        doc_index = np.repeat(np.arange(len(lengths)), lengths)
        counts = np.bincount(
            doc_index * len(distinct_rows) + row_index,
            weights=None if self.weights is None else self.weights[ids],
            minlength=len(lengths) * len(distinct_rows),
        ).astype(np.float32)
        return counts.reshape(len(lengths), len(distinct_rows)) @ self.embeddings[distinct_rows]

    def embed(
        self,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        """
        Encode a list of documents into list of embeddings.

        Args:
            documents: Iterator of documents or single document to embed
            batch_size: Batch size for encoding
            parallel:
                If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
                If 0, use all available cores.
                If None, don't use data-parallel processing.

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

    def _embed_documents(
        self,
        model_name: str,
        cache_dir: str,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
        local_files_only: bool = False,
        specific_model_path: str | None = None,
    ) -> Iterable[NumpyArray]:
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
                "local_files_only": local_files_only,
                "specific_model_path": specific_model_path,
            }
            pool = ParallelWorkerPool(
                num_workers=parallel or 1,
                worker=self._get_worker_class(),
                start_method=start_method,
            )
            for batch in pool.ordered_map(iter_batch(documents, batch_size), **params):
                yield from batch

    @classmethod
    def _get_worker_class(cls) -> Type["StaticTextEmbeddingWorker"]:
        return StaticTextEmbeddingWorker

    def token_count(
        self, texts: str | Iterable[str], batch_size: int = 1024, **kwargs: Any
    ) -> int:
        """Returns the number of tokens after truncation, unknown tokens included."""
        self._ensure_model()
        texts = [texts] if isinstance(texts, str) else texts
        return sum(int(self._token_ids(batch)[1].sum()) for batch in iter_batch(texts, batch_size))


class StaticTextEmbeddingWorker(Worker):
    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs: Any,
    ):
        self.model = self.init_embedding(model_name, cache_dir, **kwargs)

    @classmethod
    def start(cls, model_name: str, cache_dir: str, **kwargs: Any) -> "StaticTextEmbeddingWorker":
        return cls(model_name=model_name, cache_dir=cache_dir, **kwargs)

    def process(self, items: Iterable[tuple[int, Any]]) -> Iterable[tuple[int, NumpyArray]]:
        for idx, batch in items:
            yield idx, self.model.raw_embed(batch)

    @staticmethod
    def init_embedding(model_name: str, cache_dir: str, **kwargs: Any) -> StaticTextEmbedding:
        return StaticTextEmbedding(model_name=model_name, cache_dir=cache_dir, **kwargs)
