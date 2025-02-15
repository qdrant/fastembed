from dataclasses import dataclass, field
from typing import Optional, Any


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
class BaseModelDescription:
    model: str
    sources: ModelSource
    model_file: str
    description: str
    license: str
    size_in_GB: float
    additional_files: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DenseModelDescription(BaseModelDescription):
    dim: Optional[int] = None
    tasks: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        assert self.dim is not None, "dim is required for dense model description"


@dataclass(frozen=True)
class SparseModelDescription(BaseModelDescription):
    requires_idf: Optional[bool] = None
    vocab_size: Optional[int] = None
