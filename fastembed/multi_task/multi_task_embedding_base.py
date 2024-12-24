from dataclasses import dataclass
from typing import Iterable, Optional, Union, Dict, List

import numpy as np

from fastembed.common.model_management import ModelManagement


@dataclass
class MultiTaskEmbedding:
    embedding: np.ndarray
    task_type: str
    dimension: Optional[int] = None

    def __post_init__(self):
        if self.dimension is None:
            self.dimension = self.embedding.shape[-1]
        elif self.embedding.shape[-1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension {self.dimension}, got {self.embedding.shape[-1]}"
            )

    def as_object(self) -> Dict[str, Union[np.ndarray, str, Optional[int]]]:
        return {
            "embedding": self.embedding,
            "task_type": self.task_type,
            "dimension": self.dimension,
        }

    def as_dict(self) -> Dict[str, Union[List[float], str, Optional[int]]]:
        return {
            "embedding": self.embedding.tolist(),
            "task_type": self.task_type,
            "dimension": self.dimension,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Union[List[float], str, int]]) -> "MultiTaskEmbedding":
        embedding = np.array(data["embedding"])
        task_type = str(data["task_type"])

        dimension = data.get("dimension")
        if dimension is not None:
            if isinstance(dimension, int):
                dimension = int(dimension)
            else:
                raise ValueError(
                    f"Invalid type for dimension: expected int or None, got {type(dimension).__name__}"
                )

        return cls(embedding=embedding, task_type=task_type, dimension=dimension)


class MultiTaskTextEmbeddingBase(ModelManagement):
    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.threads = threads
        self._local_files_only = kwargs.pop("local_files_only", False)

    def task_embed(
        self,
        documents: Union[str, Iterable[str]],
        task_type: str,
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs,
    ) -> Iterable[MultiTaskEmbedding]:
        raise NotImplementedError()
