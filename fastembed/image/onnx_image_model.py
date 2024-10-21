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
from fastembed.parallel_processor import ParallelWorkerPool

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

    def _load_onnx_model(
        self,
        model_dir: Path,
        model_file: str,
        threads: Optional[int],
        providers: Optional[Sequence[OnnxProvider]] = None,
        cuda: bool = False,
        device_id: Optional[int] = None,
    ) -> None:
        super()._load_onnx_model(
            model_dir=model_dir,
            model_file=model_file,
            threads=threads,
            providers=providers,
            cuda=cuda,
            device_id=device_id,
        )
        self.processor = load_preprocessor(model_dir=model_dir)

    def load_onnx_model(self) -> None:
        raise NotImplementedError("Subclasses must implement this method")

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

    def _embed_images(
        self,
        model_name: str,
        cache_dir: str,
        images: ImageInput,
        batch_size: int = 256,
        parallel: Optional[int] = None,
        providers: Optional[Sequence[OnnxProvider]] = None,
        cuda: bool = False,
        device_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> Iterable[T]:
        is_small = False

        if isinstance(images, (str, Path, Image.Image)):
            images = [images]
            is_small = True

        if isinstance(images, list) and len(images) < batch_size:
            is_small = True

        if parallel is None or is_small:
            if not hasattr(self, "model") or self.model is None:
                self.load_onnx_model()

            for batch in iter_batch(images, batch_size):
                yield from self._post_process_onnx_output(self.onnx_embed(batch))
        else:
            if parallel == 0:
                parallel = os.cpu_count()

            start_method = "forkserver" if "forkserver" in get_all_start_methods() else "spawn"
            params = {
                "model_name": model_name,
                "cache_dir": cache_dir,
                "providers": providers,
                **kwargs,
            }

            pool = ParallelWorkerPool(
                num_workers=parallel or 1,
                worker=self._get_worker_class(),
                cuda=cuda,
                device_ids=device_ids,
                start_method=start_method,
            )
            for batch in pool.ordered_map(iter_batch(images, batch_size), **params):
                yield from self._post_process_onnx_output(batch)


class ImageEmbeddingWorker(EmbeddingWorker):
    def process(self, items: Iterable[Tuple[int, Any]]) -> Iterable[Tuple[int, Any]]:
        for idx, batch in items:
            embeddings = self.model.onnx_embed(batch)
            yield idx, embeddings
