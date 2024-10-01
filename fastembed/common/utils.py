import os
import tempfile
from itertools import islice
from pathlib import Path
from typing import Generator, Iterable, Optional, Union
import unicodedata
import sys
import numpy as np
import re
from typing import Set


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


def get_all_punctuation() -> Set[str]:
    return set(
        chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")
    )


def remove_non_alphanumeric(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
