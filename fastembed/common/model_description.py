from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass(frozen=True)
class ModelSource:
    hf: Optional[str] = None
    url: Optional[str] = None

    def __post_init__(self):
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
    size_in_GB: float
    additional_files: List[str] = field(default_factory=list)
    tasks: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class SparseModelDescription(ModelDescription):
    vocab_size: int
    requires_idf: Optional[bool] = None
    # For sparse models, override dim to always be None.
    dim: Optional[int] = None


@dataclass(frozen=True)
class CustomModelDescription(ModelDescription):
    description: str = ""
    license: str = ""
    size_in_GB: Optional[float] = None
