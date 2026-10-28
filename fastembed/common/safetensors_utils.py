import json
import math
from pathlib import Path

import numpy as np

from fastembed.common.types import NumpyArray

# safetensors dtype -> numpy dtype. BF16 has no numpy dtype and is widened to float32 on load.
_DTYPES: dict[str, str] = {
    "F64": "<f8",
    "F32": "<f4",
    "F16": "<f2",
    "I64": "<i8",
    "I32": "<i4",
    "I16": "<i2",
    "I8": "i1",
    "U8": "u1",
    "BOOL": "?",
}


def load_safetensors(path: str | Path, mmap: bool = False) -> dict[str, NumpyArray]:
    """Read every tensor of a safetensors file with numpy, without the `safetensors` package.

    The format is an 8 byte little-endian header size, a JSON header with the dtype, shape and
    byte offsets of each tensor, then the raw tensor data.

    Args:
        path: The safetensors file.
        mmap: Memory-map the tensors read-only, so loading is instant, only the rows in use take
            memory and parallel workers share the pages of one file. Otherwise the tensors are
            read into memory, which avoids page faults and keeps no file open.
    """
    path = Path(path)
    with open(path, "rb") as f:
        header_size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_size))
    header.pop("__metadata__", None)
    data_start = 8 + header_size

    tensors: dict[str, NumpyArray] = {}
    for name, info in header.items():
        start, end = info["data_offsets"]
        shape = tuple(info["shape"])
        dtype = info["dtype"]
        if dtype != "BF16" and dtype not in _DTYPES:
            raise ValueError(f"Unsupported safetensors dtype {dtype} for {name} in {path}")
        item_size = 2 if dtype == "BF16" else np.dtype(_DTYPES[dtype]).itemsize
        expected_size = math.prod(shape) * item_size
        if end < start or end - start != expected_size:
            raise ValueError(
                f"Invalid safetensors data offsets [{start}, {end}] for {name} in {path}: "
                f"expected {expected_size} bytes for shape {list(shape)}"
            )
        if end == start:
            tensors[name] = np.empty(
                shape, dtype=np.float32 if dtype == "BF16" else _DTYPES[dtype]
            )
            continue
        if dtype == "BF16":
            raw = _read(path, "<u2", data_start + start, shape, mmap)
            tensors[name] = (raw.astype(np.uint32) << 16).view(np.float32)
            continue
        tensors[name] = _read(path, _DTYPES[dtype], data_start + start, shape, mmap)
    return tensors


def _read(path: Path, dtype: str, offset: int, shape: tuple[int, ...], mmap: bool) -> NumpyArray:
    if mmap:
        return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape)
    count = int(np.prod(shape))
    return np.fromfile(path, dtype=dtype, count=count, offset=offset).reshape(shape)
