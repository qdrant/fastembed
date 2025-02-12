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
    description: str = ""
    license: str = ""
    size_in_GB: Optional[float] = None
    additional_files: list[str] = field(default_factory=list)

    def validate_info(self) -> None:
        if self.license == "":
            raise ValueError("license is required for dense model description")

        if self.description == "":
            raise ValueError("description is required for dense model description")

        if self.size_in_GB is None:
            raise ValueError("size_in_GB is required for dense model description")

    def __post_init__(self) -> None:
        self.validate_info()


@dataclass(frozen=True)
class DenseModelDescription(BaseModelDescription):
    dim: Optional[int] = None
    tasks: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        assert self.dim is not None, "dim is required for dense model description"
        self.validate_info()


@dataclass(frozen=True)
class SparseModelDescription(BaseModelDescription):
    requires_idf: Optional[bool] = None
    vocab_size: Optional[int] = None


@dataclass(frozen=True)
class CustomDenseModelDescription(DenseModelDescription):
    def __post_init__(self) -> None:
        if self.dim is None:
            raise ValueError("dim is required for custom dense model description")
        # disable self.validate_info


@dataclass(frozen=True)
class CustomSparseModelDescription(SparseModelDescription):
    def __post_init__(self) -> None:
        pass  # disable self.validate_info
