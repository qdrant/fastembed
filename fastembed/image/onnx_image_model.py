import os
from multiprocessing import get_all_start_methods
from pathlib import Path
from typing import Any, Dict, Generic, Iterable, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np

from fastembed.common.onnx_model import OnnxModel, EmbeddingWorker
from fastembed.common.utils import iter_batch
from fastembed.parallel_processor import ParallelWorkerPool

# Holds type of the embedding result
T = TypeVar("T")


class OnnxImageModel(OnnxModel[Generic[T]]):
    @classmethod
    def _get_worker_class(cls) -> Type["EmbeddingWorker"]:
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def _post_process_onnx_output(cls, output: np.ndarray) -> Iterable[T]:
        raise NotImplementedError("Subclasses must implement this method")

    def __init__(self) -> None:
        super().__init__()

    def _preprocess_onnx_input(self, onnx_input: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Preprocess the onnx input.
        """
        return onnx_input

    def load_onnx_model(self, model_dir: Path, threads: Optional[int]) -> None:
        super().load_onnx_model(model_dir, threads)

    def onnx_embed(self, images: List[Union[str, Path]]) -> Tuple[np.ndarray, np.ndarray]: ...

    def _embed_images(
        self,
        model_name: str,
        cache_dir: str,
        images: Union[Union[str, Path], Iterable[Union[str, Path]]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
    ) -> Iterable[T]:
        is_small = False

        if isinstance(images, str) or isinstance(images, Path):
            images = [images]
            is_small = True

        if isinstance(images, list):
            if len(images) < batch_size:
                is_small = True

        if parallel == 0:
            parallel = os.cpu_count()

        if parallel is None or is_small:
            for batch in iter_batch(images, batch_size):
                # open and preprocess images
                yield from self._post_process_onnx_output(self.onnx_embed(batch))
        else:
            start_method = "forkserver" if "forkserver" in get_all_start_methods() else "spawn"
            params = {
                "model_name": model_name,
                "cache_dir": cache_dir,
            }
            pool = ParallelWorkerPool(
                parallel, self._get_worker_class(), start_method=start_method
            )
            for batch in pool.ordered_map(iter_batch(images, batch_size), **params):
                yield from self._post_process_onnx_output(batch)


class ImageEmbeddingWorker(EmbeddingWorker):
    def process(self, items: Iterable[Tuple[int, Any]]) -> Iterable[Tuple[int, Any]]:
        for idx, batch in items:
            embeddings = self.model.onnx_embed(batch)
            yield idx, embeddings
