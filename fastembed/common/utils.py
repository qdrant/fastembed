import os
import sys
import re
import tempfile
import unicodedata
from pathlib import Path
from functools import lru_cache
from itertools import islice
from typing import Callable, Iterable, TypeVar

import numpy as np
from numpy.typing import NDArray

from fastembed.common.types import NumpyArray

T = TypeVar("T")


def normalize(input_array: NumpyArray, p: int = 2, dim: int = 1, eps: float = 1e-12) -> NumpyArray:
    if input_array.dtype == np.float16:
        # the sum of squares overflows float16 (max 65504) already for moderate values,
        # which turns the norm into inf and the embedding into zeros
        return normalize(input_array.astype(np.float32), p=p, dim=dim, eps=eps).astype(np.float16)

    # Calculate the Lp norm along the specified dimension
    norm = np.linalg.norm(input_array, ord=p, axis=dim, keepdims=True)
    norm = np.maximum(norm, eps)  # Avoid division by zero
    normalized_array = input_array / norm
    return normalized_array


def mean_pooling(input_array: NumpyArray, attention_mask: NDArray[np.int64]) -> NumpyArray:
    """Average the embeddings of the tokens which the attention mask marks as real.

    The sum is accumulated in float64, so the result is float64 for any input dtype,
    callers cast it back to the dtype of the model once post-processing is done.
    """
    # `where` skips the padding without materializing a (batch_size, seq_len, dim) mask
    sum_embeddings = np.sum(
        input_array, axis=1, where=attention_mask[:, :, np.newaxis].astype(bool), dtype=np.float64
    )
    sum_mask = np.sum(attention_mask, axis=1, keepdims=True)
    pooled_embeddings = sum_embeddings / np.maximum(sum_mask, 1e-9)
    return pooled_embeddings


def last_token_pooling(input_array: NumpyArray, attention_mask: NDArray[np.int64]) -> NumpyArray:
    """Take the embedding of the last non-padding token of each sequence.

    Locates the last position the attention mask marks as real, so it holds whichever
    side the tokenizer pads on.
    """
    last_token_indices = attention_mask.shape[1] - 1 - np.argmax(attention_mask[:, ::-1], axis=1)
    return input_array[np.arange(input_array.shape[0]), last_token_indices]


def iter_batch(iterable: Iterable[T], size: int) -> Iterable[list[T]]:
    """Validate the batch size immediately and consume the iterable lazily.

    >>> list(iter_batch([1,2,3,4,5], 3))
    [[1, 2, 3], [4, 5]]
    """
    if size < 1:
        raise ValueError(f"batch_size must be >= 1, got {size}")

    def batches() -> Iterable[list[T]]:
        source_iter = iter(iterable)
        while source_iter:
            b = list(islice(source_iter, size))
            if len(b) == 0:
                break
            yield b

    return batches()


def iter_length_sorted_batches(
    texts: Iterable[T],
    batch_size: int,
    token_count: Callable[[T], int],
    window_batches: int = 8,
) -> Iterable[tuple[list[int], list[T]]]:
    """Batch texts of similar length together, sorting within windows of `window_batches` batches.

    A larger window saves more padding, but reads more texts ahead and delays the first output.
    Length is the token count after truncation, so truncated texts tie and keep their order.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    offset = 0
    for window in iter_batch(texts, batch_size * window_batches):
        lengths = [token_count(text) for text in window]
        order = sorted(range(len(window)), key=lambda i: lengths[i], reverse=True)
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            yield [offset + i for i in indices], [window[i] for i in indices]
        offset += len(window)


def restore_order(indexed_outputs: Iterable[tuple[list[int], Iterable[T]]]) -> Iterable[T]:
    pending: dict[int, T] = {}
    next_index = 0
    for indices, outputs in indexed_outputs:
        pending.update(zip(indices, outputs, strict=True))
        while next_index in pending:
            yield pending.pop(next_index)
            next_index += 1


def define_cache_dir(cache_dir: str | None = None) -> Path:
    """
    Define the cache directory for fastembed
    """
    if cache_dir is None:
        default_cache_dir = os.path.join(tempfile.gettempdir(), "fastembed_cache")
        cache_path = Path(os.getenv("FASTEMBED_CACHE_PATH", default_cache_dir))
    else:
        cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    return cache_path


def get_all_punctuation() -> set[str]:
    return set(
        chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")
    )


@lru_cache(maxsize=None)
def get_all_marks() -> str:
    """Return the combining marks (Unicode category M) as regex character class ranges.

    Marks include Tamil and Devanagari vowel signs and Arabic harakat. The regex word class
    does not match them, so a pattern that only keeps word characters splits words in these
    scripts at every mark. Ranges keep the class short, which keeps matching fast.
    """
    ranges: list[str] = []
    start = None
    for i in range(sys.maxunicode + 2):
        is_mark = i <= sys.maxunicode and unicodedata.category(chr(i)).startswith("M")
        if is_mark and start is None:
            start = i
        elif not is_mark and start is not None:
            ranges.append(f"{re.escape(chr(start))}-{re.escape(chr(i - 1))}")
            start = None
    return "".join(ranges)


@lru_cache(maxsize=None)
def _non_alphanumeric_pattern() -> re.Pattern[str]:
    return re.compile(rf"[^\w\s{get_all_marks()}]")


def remove_non_alphanumeric(text: str) -> str:
    # ASCII text has no combining marks, and the plain class is faster to match.
    if text.isascii():
        return re.sub(r"[^\w\s]", " ", text)
    return _non_alphanumeric_pattern().sub(" ", text)
