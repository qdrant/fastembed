from enum import Enum
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from PIL import Image


class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


PathInput: TypeAlias = str | Path
ImageInput: TypeAlias = PathInput | Image.Image

OnnxProvider: TypeAlias = str | tuple[str, dict[Any, Any]]
NumpyArray: TypeAlias = (
    NDArray[np.float64]
    | NDArray[np.float32]
    | NDArray[np.float16]
    | NDArray[np.int8]
    | NDArray[np.int64]
    | NDArray[np.int32]
)
