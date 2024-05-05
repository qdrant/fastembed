import os
from multiprocessing import get_all_start_methods
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    Sequence,
)

import numpy as np
import onnxruntime as ort

from fastembed.common.models import load_tokenizer
from fastembed.common.utils import iter_batch
from fastembed.parallel_processor import ParallelWorkerPool, Worker


# Holds type of the embedding result
T = TypeVar("T")

OnnxProvider = Union[str, Tuple[str, Dict[Any, Any]]]


class OnnxModel(Generic[T]):
    @classmethod
    def _get_worker_class(cls) -> Type["EmbeddingWorker"]:
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def _post_process_onnx_output(cls, output: Tuple[np.ndarray, np.ndarray]) -> Iterable[T]:
        raise NotImplementedError("Subclasses must implement this method")

    def __init__(self) -> None:
        self.model = None
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
        providers: Optional[Sequence[OnnxProvider]] = None,
    ) -> None:
        model_path = model_dir / model_file

        # List of Execution Providers: https://onnxruntime.ai/docs/execution-providers

        onnx_providers = ["CPUExecutionProvider"] if providers is None else list(providers)
        available_providers = ort.get_available_providers()
        for provider in onnx_providers:
            # check providers available
            provider_name = provider if isinstance(provider, str) else provider[0]
            if provider_name not in available_providers:
                raise ValueError(
                    f"Provider {provider_name} is not available. Available providers: {available_providers}"
                )

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if threads is not None:
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = threads

        self.tokenizer = load_tokenizer(model_dir=model_dir)
        self.model = ort.InferenceSession(
            str(model_path), providers=onnx_providers, sess_options=so
        )

    def onnx_embed(self, documents: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        encoded = self.tokenizer.encode_batch(documents)
        input_ids = np.array([e.ids for e in encoded])
        attention_mask = np.array([e.attention_mask for e in encoded])

        onnx_input = {
            "input_ids": np.array(input_ids, dtype=np.int64),
            "attention_mask": np.array(attention_mask, dtype=np.int64),
            "token_type_ids": np.array(
                [np.zeros(len(e), dtype=np.int64) for e in input_ids], dtype=np.int64
            ),
        }

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


class EmbeddingWorker(Worker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
    ) -> OnnxModel:
        raise NotImplementedError()

    def __init__(
        self,
        model_name: str,
        cache_dir: str,
    ):
        self.model = self.init_embedding(model_name, cache_dir)

    @classmethod
    def start(cls, model_name: str, cache_dir: str, **kwargs: Any) -> "EmbeddingWorker":
        return cls(
            model_name=model_name,
            cache_dir=cache_dir,
        )

    def process(self, items: Iterable[Tuple[int, Any]]) -> Iterable[Tuple[int, Any]]:
        for idx, batch in items:
            embeddings, attn_mask = self.model.onnx_embed(batch)
            yield idx, (embeddings, attn_mask)
