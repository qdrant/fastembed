import os
import tempfile
from itertools import islice
from pathlib import Path
from typing import Union, Iterable, Generator, Optional


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
