import os
import sys
import re
import tempfile
import unicodedata
from pathlib import Path
from itertools import islice
from typing import Generator, Iterable, Optional, Union

import numpy as np


def normalize(input_array, p=2, dim=1, eps=1e-12) -> np.ndarray:
    # Calculate the Lp norm along the specified dimension
    norm = np.linalg.norm(input_array, ord=p, axis=dim, keepdims=True)
    norm = np.maximum(norm, eps)  # Avoid division by zero
    normalized_array = input_array / norm
    return normalized_array


def iter_batch(iterable: Union[Iterable, Generator], size: int) -> Iterable:
    """
    >>> list(iter_batch([1,2,3,4,5], 3))
    [[1, 2, 3], [4, 5]]
    """
    source_iter = iter(iterable)
    while source_iter:
        b = list(islice(source_iter, size))
        if len(b) == 0:
            break
        yield b


def define_cache_dir(cache_dir: Optional[str] = None) -> Path:
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


def remove_non_alphanumeric(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)


def average_pool(embeddings: np.ndarray, attention_masks: np.ndarray) -> np.ndarray:
    """
    Perform average pooling on the embeddings, excluding padding tokens based on attention masks.

    Args:
        embeddings (np.ndarray): The embeddings of shape (batch_size, seq_length, embedding_dim).
        attention_masks (np.ndarray): The attention masks of shape (batch_size, seq_length),
                                       where 1 indicates valid tokens and 0 indicates padding.

    Returns:
        np.ndarray: Pooled embeddings of shape (batch_size, embedding_dim).
    """
    # Ensure attention masks are floats to allow broadcasting
    attention_masks = attention_masks.astype(np.float32)

    # Calculate the sum of embeddings for valid tokens
    masked_embeddings = (
        embeddings * attention_masks[..., np.newaxis]
    )  # Apply mask along seq_length
    sum_embeddings = np.sum(masked_embeddings, axis=1)

    # Count the number of valid tokens per sequence
    token_counts = (
        np.sum(attention_masks, axis=1, keepdims=True) + 1e-9
    )  # Add small value to prevent division by zero

    # Compute the average for each sequence
    pooled_embeddings = sum_embeddings / token_counts

    return pooled_embeddings
