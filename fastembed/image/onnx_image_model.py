import os
import contextlib
from multiprocessing import get_all_start_methods
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union, Sequence

from PIL import Image
import numpy as np

from fastembed.common.models import load_preprocessor
from fastembed.common.onnx_model import OnnxModel, EmbeddingWorker, T, OnnxProvider
from fastembed.common.utils import iter_batch
from fastembed.parallel_processor import ParallelWorkerPool

# Holds type of the embedding result


class OnnxImageModel(OnnxModel[T]):
    @classmethod
    def _get_worker_class(cls) -> Type["ImageEmbeddingWorker"]:
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def _post_process_onnx_output(cls, output: np.ndarray) -> Iterable[T]:
        raise NotImplementedError("Subclasses must implement this method")

    def __init__(self) -> None:
        super().__init__()
        self.processor = None

    def _preprocess_onnx_input(self, onnx_input: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Preprocess the onnx input.
        """
        return onnx_input

    def load_onnx_model(
        self,
        model_dir: Path,
        model_file: str,
        threads: Optional[int],
        providers: Optional[Sequence[OnnxProvider]] = None,
    ) -> None:
        super().load_onnx_model(
            model_dir=model_dir, model_file=model_file, threads=threads, providers=providers
        )
        self.processor = load_preprocessor(model_dir=model_dir)

    def onnx_embed(self, images: List[Union[str, Path]]) -> np.ndarray:
        with contextlib.ExitStack():
            image_files = [Image.open(image) for image in images]
            encoded = self.processor(image_files)
        onnx_input = {"pixel_values": encoded}
        onnx_input = self._preprocess_onnx_input(onnx_input)

        model_output = self.model.run(None, onnx_input)
        embeddings = model_output[0]
        return embeddings

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
