import contextlib
import os
from multiprocessing import get_all_start_methods
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type

import numpy as np
from PIL import Image

from fastembed.common import ImageInput, OnnxProvider
from fastembed.common.onnx_model import EmbeddingWorker, OnnxModel, OnnxOutputContext, T
from fastembed.common.preprocessor_utils import load_preprocessor
from fastembed.common.utils import iter_batch
from fastembed.parallel_processor import ParallelWorkerPool, GPUParallelWorkerPool

# Holds type of the embedding result


class OnnxImageModel(OnnxModel[T]):
    @classmethod
    def _get_worker_class(cls) -> Type["ImageEmbeddingWorker"]:
        raise NotImplementedError("Subclasses must implement this method")

    def _post_process_onnx_output(self, output: OnnxOutputContext) -> Iterable[T]:
        raise NotImplementedError("Subclasses must implement this method")

    def __init__(self) -> None:
        super().__init__()
        self.processor = None

    def _preprocess_onnx_input(
        self, onnx_input: Dict[str, np.ndarray], **kwargs
    ) -> Dict[str, np.ndarray]:
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
        device_ids: Optional[List[int]] = None,
    ) -> None:
        super().load_onnx_model(
            model_dir=model_dir,
            model_file=model_file,
            threads=threads,
            providers=providers,
            device_ids=device_ids,
        )
        self.processor = load_preprocessor(model_dir=model_dir)

    def _build_onnx_input(self, encoded: np.ndarray) -> Dict[str, np.ndarray]:
        return {node.name: encoded for node in self.model.get_inputs()}

    def onnx_embed(self, images: List[ImageInput], **kwargs) -> OnnxOutputContext:
        with contextlib.ExitStack():
            image_files = [
                Image.open(image) if not isinstance(image, Image.Image) else image
                for image in images
            ]
            encoded = self.processor(image_files)
        onnx_input = self._build_onnx_input(encoded)
        onnx_input = self._preprocess_onnx_input(onnx_input)
        model_output = self.model.run(None, onnx_input)
        embeddings = model_output[0].reshape(len(images), -1)
        return OnnxOutputContext(model_output=embeddings)

    @classmethod
    def _embed_images_parallel(
        cls,
        model_name: str,
        cache_dir: str,
        images: ImageInput,
        batch_size: int = 256,
        parallel: Optional[int] = 2,
        providers: Optional[Sequence[OnnxProvider]] = None,
        device_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> Iterable[T]:
        is_small = False

        if (
            isinstance(images, str)
            or isinstance(images, Path)
            or (isinstance(images, Image.Image))
        ):
            images = [images]
            is_small = True

        if isinstance(images, list):
            if len(images) < batch_size:
                is_small = True

        if parallel == 0:
            parallel = os.cpu_count()

        if is_small:
            model = cls(model_name=model_name, cache_dir=cache_dir, providers=providers, **kwargs)
            for batch in iter_batch(images, batch_size):
                yield from model._post_process_onnx_output(model.onnx_embed(batch))
        else:
            use_multi_gpu = providers and "CUDAExecutionProvider" in providers and device_ids
            start_method = "forkserver" if "forkserver" in get_all_start_methods() else "spawn"
            params = {
                "model_name": model_name,
                "cache_dir": cache_dir,
                "providers": providers,
                **kwargs,
            }
            if use_multi_gpu:
                num_workers = min(parallel, len(device_ids))
                pool = GPUParallelWorkerPool(
                    num_workers, cls._get_worker_class(), device_ids, start_method=start_method
                )
            else:
                pool = ParallelWorkerPool(
                    parallel, cls._get_worker_class(), start_method=start_method
                )
            for batch in pool.ordered_map(iter_batch(images, batch_size), **params):
                yield from batch

    def _embed_images(
        self,
        model_name: str,
        cache_dir: str,
        images: ImageInput,
        batch_size: int = 256,
        parallel: Optional[int] = None,
        providers: Optional[Sequence[OnnxProvider]] = None,
        device_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> Iterable[T]:
        if parallel:
            yield from self._embed_images_parallel(
                model_name,
                cache_dir,
                images,
                batch_size,
                parallel,
                providers,
                device_ids,
                **kwargs,
            )
        else:
            for batch in iter_batch(images, batch_size):
                yield from self._post_process_onnx_output(self.onnx_embed(batch))


class ImageEmbeddingWorker(EmbeddingWorker):
    def process(self, items: Iterable[Tuple[int, Any]]) -> Iterable[Tuple[int, Any]]:
        for idx, batch in items:
            embeddings = self.model.onnx_embed(batch)
            yield idx, self.model._post_process_onnx_output(embeddings)
