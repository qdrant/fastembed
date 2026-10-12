import json
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


def load_safetensors(path: str | Path) -> dict[str, NumpyArray]:
    """Read every tensor of a safetensors file with numpy, without the `safetensors` package.

    The format is an 8 byte little-endian header size, a JSON header with the dtype, shape and
    byte offsets of each tensor, then the raw tensor data. Tensors are memory-mapped read-only, so
    loading is instant and parallel workers share the pages of one file.
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
        if end == start:
            tensors[name] = np.zeros(shape, dtype=np.float32)
            continue
        if info["dtype"] == "BF16":
            raw = np.memmap(path, dtype="<u2", mode="r", offset=data_start + start, shape=shape)
            tensors[name] = (raw.astype(np.uint32) << 16).view(np.float32)
            continue
        if info["dtype"] not in _DTYPES:
            raise ValueError(f"Unsupported safetensors dtype {info['dtype']} for {name} in {path}")
        tensors[name] = np.memmap(
            path, dtype=_DTYPES[info["dtype"]], mode="r", offset=data_start + start, shape=shape
        )
    return tensors
