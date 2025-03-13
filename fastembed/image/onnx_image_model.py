import contextlib
import os
from multiprocessing import get_all_start_methods
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Type, Union

import numpy as np
from PIL import Image
import onnxruntime as ort

from fastembed.image.transform.operators import Compose
from fastembed.common.types import NumpyArray
from fastembed.common import ImageInput, OnnxProvider
from fastembed.common.onnx_model import EmbeddingWorker, OnnxModel, OnnxOutputContext, T
from fastembed.common.preprocessor_utils import load_preprocessor
from fastembed.common.utils import iter_batch, is_cuda_enabled
from fastembed.parallel_processor import ParallelWorkerPool

# Holds type of the embedding result


class OnnxImageModel(OnnxModel[T]):
    @classmethod
    def _get_worker_class(cls) -> Type["ImageEmbeddingWorker[T]"]:
        raise NotImplementedError("Subclasses must implement this method")

    def _post_process_onnx_output(self, output: OnnxOutputContext) -> Iterable[T]:
        raise NotImplementedError("Subclasses must implement this method")

    def __init__(self) -> None:
        super().__init__()
        self.processor: Optional[Compose] = None

    def _preprocess_onnx_input(
        self, onnx_input: dict[str, NumpyArray], **kwargs: Any
    ) -> dict[str, NumpyArray]:
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

    def _build_onnx_input(self, encoded: NumpyArray) -> dict[str, NumpyArray]:
        input_name = self.model.get_inputs()[0].name  # type: ignore[union-attr]
        return {input_name: encoded}

    def onnx_embed(self, images: list[ImageInput], **kwargs: Any) -> OnnxOutputContext:
        with contextlib.ExitStack():
            image_files = [
                Image.open(image) if not isinstance(image, Image.Image) else image
                for image in images
            ]
            assert self.processor is not None, "Processor is not initialized"
            encoded = np.array(self.processor(image_files))
        onnx_input = self._build_onnx_input(encoded)
        onnx_input = self._preprocess_onnx_input(onnx_input)

        run_options = ort.RunOptions()
        providers = kwargs.get("providers", None)
        cuda = kwargs.get("cuda", False)
        if is_cuda_enabled(cuda, providers):
            device_id = kwargs.get("device_id", None)
            device_id = str(device_id if isinstance(device_id, int) else 0)
            # enables memory arena shrinkage, freeing unused memory after each Run() cycle.
            # helps prevent excessive memory retention, especially for dynamic workloads.
            # source: https://onnxruntime.ai/docs/get-started/with-c.html#features:~:text=Memory%20arena%20shrinkage:
            run_options.add_run_config_entry(
                "memory.enable_memory_arena_shrinkage", f"gpu:{device_id}"
            )

        model_output = self.model.run(None, onnx_input, run_options)  # type: ignore[union-attr]
        embeddings = model_output[0].reshape(len(images), -1)
        return OnnxOutputContext(model_output=embeddings)

    def _embed_images(
        self,
        model_name: str,
        cache_dir: str,
        images: Union[ImageInput, Iterable[ImageInput]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        providers: Optional[Sequence[OnnxProvider]] = None,
        cuda: bool = False,
        device_ids: Optional[list[int]] = None,
        **kwargs: Any,
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
                yield from self._post_process_onnx_output(
                    self.onnx_embed(batch, cuda=cuda, providers=providers)
                )
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
                yield from self._post_process_onnx_output(batch)  # type: ignore


class ImageEmbeddingWorker(EmbeddingWorker[T]):
    def process(
        self, items: Iterable[tuple[int, Any]], **kwargs: Any
    ) -> Iterable[tuple[int, Any]]:
        for idx, batch in items:
            embeddings = self.model.onnx_embed(batch, **kwargs)
            yield idx, embeddings
