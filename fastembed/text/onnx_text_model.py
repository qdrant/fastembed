import os
from multiprocessing import get_all_start_methods
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

import numpy as np

from fastembed.common.models import load_tokenizer
from fastembed.common.onnx_model import OnnxModel, EmbeddingWorker, T
from fastembed.common.utils import iter_batch
from fastembed.parallel_processor import ParallelWorkerPool


class OnnxTextModel(OnnxModel[T]):
    @classmethod
    def _get_worker_class(cls) -> Type["TextEmbeddingWorker"]:
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def _post_process_onnx_output(cls, output: Tuple[np.ndarray, np.ndarray]) -> Iterable[T]:
        raise NotImplementedError("Subclasses must implement this method")

    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = None

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
    ) -> None:
        super().load_onnx_model(model_dir=model_dir, model_file=model_file, threads=threads)
        self.tokenizer = load_tokenizer(model_dir=model_dir)

    def onnx_embed(self, documents: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        encoded = self.tokenizer.encode_batch(documents)
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

        onnx_input = self._preprocess_onnx_input(onnx_input)

        model_output = self.model.run(None, onnx_input)
        embeddings = model_output[0]
        return embeddings, attention_mask

    def _embed_documents(
        self,
        model_name: str,
        cache_dir: str,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
    ) -> Iterable[T]:
        is_small = False

        if isinstance(documents, str):
            documents = [documents]
            is_small = True

        if isinstance(documents, list):
            if len(documents) < batch_size:
                is_small = True

        if parallel == 0:
            parallel = os.cpu_count()

        if parallel is None or is_small:
            for batch in iter_batch(documents, batch_size):
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
            for batch in pool.ordered_map(iter_batch(documents, batch_size), **params):
                yield from self._post_process_onnx_output(batch)


class TextEmbeddingWorker(EmbeddingWorker):
    def process(self, items: Iterable[Tuple[int, Any]]) -> Iterable[Tuple[int, Any]]:
        for idx, batch in items:
            embeddings, attn_mask = self.model.onnx_embed(batch)
            yield idx, (embeddings, attn_mask)
