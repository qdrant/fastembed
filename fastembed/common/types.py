import os
import sys
from typing import IO, Any, Dict, Iterable, Tuple, Union

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias


DataInput: TypeAlias = IO[bytes]
PathInput: TypeAlias = Union[str, os.PathLike]
ImageInput: TypeAlias = Union[PathInput, Iterable[PathInput], DataInput, Iterable[DataInput]]

OnnxProvider: TypeAlias = Union[str, Tuple[str, Dict[Any, Any]]]
