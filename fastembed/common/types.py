import os
import sys
from typing import Any, Dict, Iterable, Tuple, Union

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias


PathInput: TypeAlias = Union[str, os.PathLike]
ImageInput: TypeAlias = Union[PathInput, Iterable[PathInput]]

OnnxProvider: TypeAlias = Union[str, Tuple[str, Dict[Any, Any]]]
