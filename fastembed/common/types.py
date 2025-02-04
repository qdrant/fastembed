from pathlib import Path
import sys
from PIL import Image
from dataclasses import dataclass
from typing import Any, Iterable, Union, Optional, TypeVar

import numpy as np
from numpy.typing import NDArray

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

# Holds type of the embedding result
T = TypeVar("T")

PathInput: TypeAlias = Union[str, Path]
PilInput: TypeAlias = Union[Image.Image, Iterable[Image.Image]]
ImageInput: TypeAlias = Union[PathInput, Iterable[PathInput], PilInput]

OnnxProvider: TypeAlias = Union[str, tuple[str, dict[Any, Any]]]

NumpyArray = Union[
    NDArray[np.float32],
    NDArray[np.float16],
    NDArray[np.int8],
    NDArray[np.int64],
    NDArray[np.int32],
]


@dataclass
class OnnxOutputContext:
    model_output: NumpyArray
    attention_mask: Optional[NumpyArray] = None
    input_ids: Optional[NumpyArray] = None
