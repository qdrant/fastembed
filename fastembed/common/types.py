import os
from typing import Union, TypeAlias, Iterable


PathInput: TypeAlias = Union[str, os.PathLike]
ImageInput: TypeAlias = Union[PathInput, Iterable[PathInput]]
