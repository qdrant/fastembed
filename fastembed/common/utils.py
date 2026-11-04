import os
import sys
import re
import tempfile
import unicodedata
from pathlib import Path
from functools import cache
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


def _is_non_linguistic_mark(code: int, category: str) -> bool:
    """True for emoji/symbol marks that should not stay inside a word token.

    Variation selectors (U+FE00–U+FE0F, U+E0100–U+E01EF) and enclosing marks
    such as the keycap U+20E3 decorate digits and emoji; keeping them after a
    digit glues the next word onto that digit (``1\\ufe0f\\u20e3hello``).
    """
    if category == "Me":
        return True
    return 0xFE00 <= code <= 0xFE0F or 0xE0100 <= code <= 0xE01EF


def normalize_for_preserved_marks(text: str) -> str:
    """NFC-normalize, fold U+0130, lowercase, and NFC-normalize again.

    NFC first composes ``I`` + U+0307 into U+0130 so both spellings of dotted
    capital I fold to ASCII ``i``. The replacement happens *before* lowercasing
    so Python does not emit ``i`` + U+0307 from U+0130. Dots already present
    in lowercase input (for example Lithuanian ``i\\u0307``) are left intact
    because they do not compose to U+0130. The final NFC pass unifies other
    canonically equivalent mark orders.
    """
    text = unicodedata.normalize("NFC", text).replace("\u0130", "i").lower()
    return unicodedata.normalize("NFC", text)


@cache
def get_all_marks() -> str:
    """Return linguistic combining marks as regex character class ranges.

    Includes Tamil and Devanagari vowel signs and Arabic harakat (Unicode
    category M), but not variation selectors or enclosing marks, which are
    treated as non-word. The regex word class does not match these marks, so
    a pattern that only keeps word characters splits words in these scripts
    at every mark. Ranges keep the class short, which keeps matching fast.
    """
    ranges: list[str] = []
    start = None
    for i in range(sys.maxunicode + 2):
        if i <= sys.maxunicode:
            category = unicodedata.category(chr(i))
            is_mark = category.startswith("M") and not _is_non_linguistic_mark(i, category)
        else:
            is_mark = False
        if is_mark and start is None:
            start = i
        elif not is_mark and start is not None:
            ranges.append(f"{re.escape(chr(start))}-{re.escape(chr(i - 1))}")
            start = None
    return "".join(ranges)


@cache
def _non_alphanumeric_pattern() -> re.Pattern[str]:
    return re.compile(rf"[^\w\s{get_all_marks()}]")


def _strip_unattached_combining_marks(text: str) -> str:
    """Drop combining marks that are not linguistic diacritics on a letter or digit.

    Marks that follow a letter or digit, including further marks on the same
    base, are kept. Leading marks, marks after symbols or whitespace, and
    emoji/keycap marks (variation selectors, enclosing keycap) are removed.
    """
    kept: list[str] = []
    attached_to_alnum = False
    for char in text:
        category = unicodedata.category(char)
        if category.startswith("M"):
            if attached_to_alnum and not _is_non_linguistic_mark(ord(char), category):
                kept.append(char)
            continue
        kept.append(char)
        attached_to_alnum = char.isalnum()
    return "".join(kept)


def remove_non_alphanumeric(text: str, preserve_combining_marks: bool = False) -> str:
    """Replace characters that are not word characters or whitespace with spaces.

    The default path is the legacy ``[^\\w\\s]`` pattern. Combining marks
    (Unicode category M) are not word characters, so ``cafe\\u0301`` becomes
    ``cafe``. Existing BM25 indexes were built with that behavior.

    Pass ``preserve_combining_marks=True`` to NFC-normalize and keep combining
    marks that follow a letter or digit, which avoids splitting words in
    scripts such as Tamil, Devanagari, and Arabic. Orphan marks, variation
    selectors, and enclosing keycaps are still stripped. Re-embed collections
    after enabling this so queries match the new tokens.
    """
    # ASCII text has no combining marks, and the plain class is faster to match.
    if text.isascii() or not preserve_combining_marks:
        return re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = unicodedata.normalize("NFC", text)
    return _strip_unattached_combining_marks(_non_alphanumeric_pattern().sub(" ", text))
