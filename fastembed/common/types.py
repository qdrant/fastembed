from pathlib import Path
import sys
from PIL import Image
from typing import Any, Union
import numpy as np
from numpy.typing import NDArray

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias


PathInput: TypeAlias = Union[str, Path]
ImageInput: TypeAlias = Union[PathInput, Image.Image]

OnnxProvider: TypeAlias = Union[str, tuple[str, dict[Any, Any]]]
NumpyArray = Union[
    NDArray[np.float64],
    NDArray[np.float32],
    NDArray[np.float16],
    NDArray[np.int8],
    NDArray[np.int64],
    NDArray[np.int32],
]
