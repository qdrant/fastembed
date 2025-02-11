from dataclasses import dataclass, field, InitVar
from typing import Optional, List, Dict


@dataclass(frozen=True)
class ModelSource:
    hf: Optional[str] = None
    url: Optional[str] = None

    def __post_init__(self) -> None:
        if self.hf is None and self.url is None:
            raise ValueError(
                f"At least one source should be set, current sources: hf={self.hf}, url={self.url}"
            )


@dataclass(frozen=True)
class ModelDescription:
    model: str
    sources: ModelSource
    model_file: str
    dim: Optional[int]

    description: str
    license: str
    size_in_GB: Optional[float]
    additional_files: List[str] = field(default_factory=list)
    tasks: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class MultimodalModelDescription(ModelDescription):
    dim: int


@dataclass(frozen=True)
class SparseModelDescription(ModelDescription):
    _vocab_size: InitVar[Optional[int]] = None
    _requires_idf: InitVar[Optional[bool]] = None

    vocab_size: int = field(init=False)
    requires_idf: Optional[bool] = field(init=False, default=None)
    dim: Optional[int] = field(default=None, init=False)

    def __init__(
        self,
        *,
        model: str,
        sources: ModelSource,
        model_file: str,
        description: str,
        license: str,
        size_in_GB: Optional[float],
        dim: Optional[int] = None,
        additional_files: Optional[List[str]] = None,
        tasks: Optional[Dict[str, int]] = None,
        vocab_size: int,
        requires_idf: Optional[bool] = None,
    ):
        # Call the parent initializer with the fields it needs.
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "model_file", model_file)
        object.__setattr__(self, "dim", dim if dim else None)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "license", license)
        object.__setattr__(self, "size_in_GB", size_in_GB)
        object.__setattr__(
            self, "additional_files", additional_files if additional_files is not None else []
        )
        object.__setattr__(self, "tasks", tasks if tasks is not None else {})
        # Set new fields.
        object.__setattr__(self, "vocab_size", vocab_size)
        object.__setattr__(self, "requires_idf", requires_idf)


@dataclass(frozen=True)
class CustomModelDescription(ModelDescription):
    description: str = ""
    license: str = ""
    size_in_GB: Optional[float] = None
