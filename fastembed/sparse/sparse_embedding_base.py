from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Union

import numpy as np

from fastembed.common.model_management import ModelManagement


@dataclass
class SparseEmbedding:
    values: np.ndarray
    indices: np.ndarray

    def as_object(self) -> Dict[str, np.ndarray]:
        return {
            "values": self.values,
            "indices": self.indices,
        }

    def as_dict(self) -> Dict[int, float]:
        return {i: v for i, v in zip(self.indices, self.values)}


class SparseTextEmbeddingBase(ModelManagement):
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

    def embed(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs,
    ) -> Iterable[SparseEmbedding]:
        raise NotImplementedError()
