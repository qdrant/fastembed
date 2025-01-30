import os
from multiprocessing import get_all_start_methods
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Type, Union

import numpy as np
from numpy.typing import NDArray
from tokenizers import Encoding

from fastembed.common.onnx_model import (
    EmbeddingWorker,
    OnnxModel,
    OnnxOutputContext,
    OnnxProvider,
)
from fastembed.common.preprocessor_utils import load_tokenizer
from fastembed.common.utils import iter_batch
from fastembed.parallel_processor import ParallelWorkerPool


class OnnxCrossEncoderModel(OnnxModel[float]):
    ONNX_OUTPUT_NAMES: Optional[list[str]] = None

    @classmethod
    def _get_worker_class(cls) -> Type["TextRerankerWorker"]:
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
        self.tokenizer, _ = load_tokenizer(model_dir=model_dir)

    def tokenize(self, pairs: list[tuple[str, str]], **_: Any) -> list[Encoding]:
        return self.tokenizer.encode_batch(pairs)

    def _build_onnx_input(
        self, tokenized_input
    ) -> dict[str, NDArray[Union[np.float32, np.int64]]]:
        input_names = {node.name for node in self.model.get_inputs()}
        inputs = {
            "input_ids": np.array([enc.ids for enc in tokenized_input], dtype=np.int64),
        }
        if "token_type_ids" in input_names:
            inputs["token_type_ids"] = np.array(
                [enc.type_ids for enc in tokenized_input], dtype=np.int64
            )
        if "attention_mask" in input_names:
            inputs["attention_mask"] = np.array(
                [enc.attention_mask for enc in tokenized_input], dtype=np.int64
            )
        return inputs

    def onnx_embed(self, query: str, documents: list[str], **kwargs: Any) -> OnnxOutputContext:
        pairs = [(query, doc) for doc in documents]
        return self.onnx_embed_pairs(pairs, **kwargs)

    def onnx_embed_pairs(self, pairs: list[tuple[str, str]], **kwargs: Any) -> OnnxOutputContext:
        tokenized_input = self.tokenize(pairs, **kwargs)
        inputs = self._build_onnx_input(tokenized_input)
        onnx_input = self._preprocess_onnx_input(inputs, **kwargs)
        outputs = self.model.run(self.ONNX_OUTPUT_NAMES, onnx_input)
        relevant_output = outputs[0]
        scores = relevant_output[:, 0]
        return OnnxOutputContext(model_output=scores)

    def _rerank_documents(
        self, query: str, documents: Iterable[str], batch_size: int, **kwargs: Any
    ) -> Iterable[float]:
        if not hasattr(self, "model") or self.model is None:
            self.load_onnx_model()
        for batch in iter_batch(documents, batch_size):
            yield from self._post_process_onnx_output(self.onnx_embed(query, batch, **kwargs))

    def _rerank_pairs(
        self,
        model_name: str,
        cache_dir: str,
        pairs: Iterable[tuple[str, str]],
        batch_size: int,
        parallel: Optional[int] = None,
        providers: Optional[Sequence[OnnxProvider]] = None,
        cuda: bool = False,
        device_ids: Optional[list[int]] = None,
        **kwargs: Any,
    ) -> Iterable[float]:
        is_small = False

        if isinstance(pairs, tuple):
            pairs = [pairs]
            is_small = True

        if isinstance(pairs, list):
            if len(pairs) < batch_size:
                is_small = True

        if parallel is None or is_small:
            if not hasattr(self, "model") or self.model is None:
                self.load_onnx_model()
            for batch in iter_batch(pairs, batch_size):
                yield from self._post_process_onnx_output(self.onnx_embed_pairs(batch, **kwargs))
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
            for batch in pool.ordered_map(iter_batch(pairs, batch_size), **params):
                yield from self._post_process_onnx_output(batch)

    def _post_process_onnx_output(self, output: OnnxOutputContext) -> Iterable[float]:
        raise NotImplementedError("Subclasses must implement this method")

    def _preprocess_onnx_input(
        self, onnx_input: dict[str, np.ndarray], **kwargs: Any
    ) -> dict[str, np.ndarray]:
        """
        Preprocess the onnx input.
        """
        return onnx_input


class TextRerankerWorker(EmbeddingWorker):
    def process(self, items: Iterable[tuple[int, Any]]) -> Iterable[tuple[int, Any]]:
        for idx, batch in items:
            onnx_output = self.model.onnx_embed_pairs(batch)
            yield idx, onnx_output
