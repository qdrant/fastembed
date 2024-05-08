import os
from typing import Union, TypeAlias, Iterable, Tuple, Dict, Any

PathInput: TypeAlias = Union[str, os.PathLike]
ImageInput: TypeAlias = Union[PathInput, Iterable[PathInput]]

OnnxProvider: TypeAlias = Union[str, Tuple[str, Dict[Any, Any]]]
