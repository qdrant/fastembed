import os
import sys
import re
import tempfile
import unicodedata
import uuid
from pathlib import Path
from itertools import islice
from typing import Callable, Iterable, TypeVar

from loguru import logger

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


def _is_writable_dir(path: Path) -> bool:
    """
    Check if a directory can be written to by creating and removing a temporary probe file.
    Uses a UUID-suffixed filename to avoid collisions between concurrent processes or threads.
    """
    probe_file = path / f".fastembed_write_test_{uuid.uuid4().hex}"
    try:
        probe_file.touch(exist_ok=False)
        return True
    except (OSError, PermissionError):
        return False
    finally:
        try:
            probe_file.unlink(missing_ok=True)
        except OSError:
            pass


def get_default_cache_dir() -> Path:
    """
    Get the default cache directory for fastembed.

    Priority:
    1. FASTEMBED_CACHE_PATH environment variable if set.
    2. Platform-specific persistent user cache directory:
       - Windows: %LOCALAPPDATA%/fastembed_cache or ~/.cache/fastembed_cache
       - macOS: ~/Library/Caches/fastembed_cache
       - Linux/other: $XDG_CACHE_HOME/fastembed_cache or ~/.cache/fastembed_cache
    3. Fallback to system temporary directory (tempfile.gettempdir()/fastembed_cache)
       if the persistent directory cannot be created or written to.
    """
    env_path = os.getenv("FASTEMBED_CACHE_PATH")
    if env_path:
        return Path(env_path)

    try:
        if sys.platform == "win32":
            local_appdata = os.getenv("LOCALAPPDATA")
            if local_appdata and Path(local_appdata).is_absolute():
                candidate = Path(local_appdata) / "fastembed_cache"
            else:
                candidate = Path.home() / ".cache" / "fastembed_cache"
        elif sys.platform == "darwin":
            candidate = Path.home() / "Library" / "Caches" / "fastembed_cache"
        else:
            xdg_cache_home = os.getenv("XDG_CACHE_HOME")
            if xdg_cache_home and Path(xdg_cache_home).is_absolute():
                candidate = Path(xdg_cache_home) / "fastembed_cache"
            else:
                candidate = Path.home() / ".cache" / "fastembed_cache"

        candidate.mkdir(parents=True, exist_ok=True)
        if not _is_writable_dir(candidate):
            raise PermissionError(f"Directory {candidate} exists but is not writable")
        return candidate
    except (OSError, PermissionError) as exc:
        fallback = Path(tempfile.gettempdir()) / "fastembed_cache"
        logger.warning(
            f"Failed to initialize persistent cache directory '{candidate}': {exc}. "
            f"Falling back to temporary cache directory at '{fallback}'."
        )
        try:
            fallback.mkdir(parents=True, exist_ok=True)
            if not _is_writable_dir(fallback):
                raise PermissionError(f"Fallback cache directory '{fallback}' is not writable.")
            return fallback
        except (OSError, PermissionError) as fallback_exc:
            raise PermissionError(
                f"Failed to create or write to default cache directory: neither persistent "
                f"location ('{candidate}') nor fallback location ('{fallback}') is usable."
            ) from fallback_exc


def define_cache_dir(cache_dir: str | None = None) -> Path:
    """
    Define the cache directory for fastembed
    """
    if cache_dir is None:
        cache_path = get_default_cache_dir()
    else:
        cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    return cache_path


def get_all_punctuation() -> set[str]:
    return set(
        chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")
    )


def remove_non_alphanumeric(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
