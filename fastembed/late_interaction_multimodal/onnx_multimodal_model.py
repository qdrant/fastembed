import contextlib
import os
from multiprocessing import get_all_start_methods
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Type, Union, get_args

import numpy as np
from PIL import Image
from tokenizers import Encoding

from fastembed.common import OnnxProvider, ImageInput
from fastembed.common.onnx_model import EmbeddingWorker, OnnxModel, OnnxOutputContext, T
from fastembed.common.preprocessor_utils import load_tokenizer, load_preprocessor
from fastembed.common.types import NumpyArray
from fastembed.common.utils import iter_batch
from fastembed.parallel_processor import ParallelWorkerPool


class OnnxMultimodalModel(OnnxModel[T]):
    ONNX_OUTPUT_NAMES: Optional[list[str]] = None

    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = None
        self.processor = None
        self.special_token_to_id = {}

    def _preprocess_onnx_text_input(
        self, onnx_input: dict[str, NumpyArray], **kwargs: Any
    ) -> dict[str, NumpyArray]:
        """
        Preprocess the onnx input.
        """
        return onnx_input

    def _preprocess_onnx_image_input(
        self, onnx_input: dict[str, NumpyArray], **kwargs: Any
    ) -> dict[str, NumpyArray]:
        """
        Preprocess the onnx input.
        """
        return onnx_input

    @classmethod
    def _get_text_worker_class(cls) -> Type["TextEmbeddingWorker"]:
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def _get_image_worker_class(cls) -> Type["ImageEmbeddingWorker"]:
        raise NotImplementedError("Subclasses must implement this method")

    def _post_process_onnx_image_output(self, output: OnnxOutputContext) -> Iterable[T]:
        raise NotImplementedError("Subclasses must implement this method")

    def _post_process_onnx_text_output(self, output: OnnxOutputContext) -> Iterable[T]:
        raise NotImplementedError("Subclasses must implement this method")

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
        assert self.tokenizer is not None
        self.tokenizer, self.special_token_to_id = load_tokenizer(model_dir=model_dir)
        self.processor = load_preprocessor(model_dir=model_dir)

    def load_onnx_model(self) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    def tokenize(self, documents: list[str], **kwargs: Any) -> list[Encoding]:
        return self.tokenizer.encode_batch(documents)

    def onnx_embed_text(
        self,
        documents: list[str],
        **kwargs: Any,
    ) -> OnnxOutputContext:
        encoded = self.tokenize(documents, **kwargs)
        input_ids = np.array([e.ids for e in encoded])
        attention_mask = np.array([e.attention_mask for e in encoded])
        input_names = {node.name for node in self.model.get_inputs()}
        onnx_input = {
            "input_ids": np.array(input_ids, dtype=np.int64),
        }
        if "attention_mask" in input_names:
            onnx_input["attention_mask"] = np.array(attention_mask, dtype=np.int64)
        if "token_type_ids" in input_names:
            onnx_input["token_type_ids"] = np.array(
                [np.zeros(len(e), dtype=np.int64) for e in input_ids], dtype=np.int64
            )

        onnx_input = self._preprocess_onnx_text_input(onnx_input, **kwargs)
        model_output = self.model.run(self.ONNX_OUTPUT_NAMES, onnx_input)  # type: ignore
        return OnnxOutputContext(
            model_output=model_output[0],
            attention_mask=onnx_input.get("attention_mask", attention_mask),
            input_ids=onnx_input.get("input_ids", input_ids),
        )

    def _embed_documents(
        self,
        model_name: str,
        cache_dir: str,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        providers: Optional[Sequence[OnnxProvider]] = None,
        cuda: bool = False,
        device_ids: Optional[list[int]] = None,
        **kwargs: Any,
    ) -> Iterable[T]:
        is_small = False

        if isinstance(documents, str):
            documents = [documents]
            is_small = True

        if isinstance(documents, list):
            if len(documents) < batch_size:
                is_small = True

        if parallel is None or is_small:
            if not hasattr(self, "model") or self.model is None:
                self.load_onnx_model()
            for batch in iter_batch(documents, batch_size):
                yield from self._post_process_onnx_text_output(self.onnx_embed_text(batch))
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
                worker=self._get_text_worker_class(),
                cuda=cuda,
                device_ids=device_ids,
                start_method=start_method,
            )
            for batch in pool.ordered_map(iter_batch(documents, batch_size), **params):
                yield from self._post_process_onnx_text_output(batch)

    def _build_onnx_image_input(self, encoded: NumpyArray) -> dict[str, NumpyArray]:
        input_name = self.model.get_inputs()[0].name  # type: ignore
        return {input_name: encoded}

    def onnx_embed_image(self, images: list[ImageInput], **kwargs: Any) -> OnnxOutputContext:
        with contextlib.ExitStack():
            image_files = [
                Image.open(image) if not isinstance(image, Image.Image) else image
                for image in images
            ]
            encoded = self.processor(image_files)
        onnx_input = self._build_onnx_image_input(encoded)
        onnx_input = self._preprocess_onnx_image_input(onnx_input, **kwargs)
        model_output = self.model.run(None, onnx_input)
        embeddings = model_output[0].reshape(len(images), -1)
        return OnnxOutputContext(model_output=embeddings)

    def _embed_images(
        self,
        model_name: str,
        cache_dir: str,
        images: Union[Iterable[ImageInput], ImageInput],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        providers: Optional[Sequence[OnnxProvider]] = None,
        cuda: bool = False,
        device_ids: Optional[list[int]] = None,
        **kwargs: Any,
    ) -> Iterable[T]:
        is_small = False

        if isinstance(images, get_args(ImageInput)):
            images = [images]
            is_small = True

        if isinstance(images, list) and len(images) < batch_size:
            is_small = True

        if parallel is None or is_small:
            if not hasattr(self, "model") or self.model is None:
                self.load_onnx_model()

            for batch in iter_batch(images, batch_size):
                yield from self._post_process_onnx_image_output(self.onnx_embed_image(batch))
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
                worker=self._get_image_worker_class(),
                cuda=cuda,
                device_ids=device_ids,
                start_method=start_method,
            )
            for batch in pool.ordered_map(iter_batch(images, batch_size), **params):
                yield from self._post_process_onnx_image_output(batch)


class TextEmbeddingWorker(EmbeddingWorker):
    def process(self, items: Iterable[tuple[int, Any]]) -> Iterable[tuple[int, Any]]:
        for idx, batch in items:
            onnx_output = self.model.onnx_embed_text(batch)
            yield idx, onnx_output


class ImageEmbeddingWorker(EmbeddingWorker):
    def process(self, items: Iterable[tuple[int, Any]]) -> Iterable[tuple[int, Any]]:
        for idx, batch in items:
            embeddings = self.model.onnx_embed_image(batch)
            yield idx, embeddings
