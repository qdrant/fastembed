import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Iterable, Optional, Sequence, Type, TypeVar

import numpy as np
import onnxruntime as ort

from numpy.typing import NDArray
from tokenizers import Tokenizer

from fastembed.common.types import OnnxProvider, NumpyArray
from fastembed.parallel_processor import Worker

# Holds type of the embedding result
T = TypeVar("T")


@dataclass
class OnnxOutputContext:
    model_output: NumpyArray
    attention_mask: Optional[NDArray[np.int64]] = None
    input_ids: Optional[NDArray[np.int64]] = None


class OnnxModel(Generic[T]):
    @classmethod
    def _get_worker_class(cls) -> Type["EmbeddingWorker[T]"]:
        raise NotImplementedError("Subclasses must implement this method")

    def _post_process_onnx_output(self, output: OnnxOutputContext, **kwargs: Any) -> Iterable[T]:
        """Post-process the ONNX model output to convert it into a usable format.

        Args:
            output (OnnxOutputContext): The raw output from the ONNX model.
            **kwargs: Additional keyword arguments that may be needed by specific implementations.

        Returns:
            Iterable[T]: Post-processed output as an iterable of type T.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def __init__(self) -> None:
        self.model: Optional[ort.InferenceSession] = None
        self.tokenizer: Optional[Tokenizer] = None

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
        model_path = model_dir / model_file
        # List of Execution Providers: https://onnxruntime.ai/docs/execution-providers

        if cuda and providers is not None:
            warnings.warn(
                f"`cuda` and `providers` are mutually exclusive parameters, cuda: {cuda}, providers: {providers}",
                category=UserWarning,
                stacklevel=6,
            )

        if providers is not None:
            onnx_providers = list(providers)
        elif cuda:
            if device_id is None:
                onnx_providers = ["CUDAExecutionProvider"]
            else:
                onnx_providers = [("CUDAExecutionProvider", {"device_id": device_id})]
        else:
            onnx_providers = ["CPUExecutionProvider"]

        available_providers = ort.get_available_providers()
        requested_provider_names: list[str] = []
        for provider in onnx_providers:
            # check providers available
            provider_name = provider if isinstance(provider, str) else provider[0]
            requested_provider_names.append(provider_name)
            if provider_name not in available_providers:
                raise ValueError(
                    f"Provider {provider_name} is not available. Available providers: {available_providers}"
                )

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if threads is not None:
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = threads

        self.model = ort.InferenceSession(
            str(model_path), providers=onnx_providers, sess_options=so
        )
        if "CUDAExecutionProvider" in requested_provider_names:
            assert self.model is not None
            current_providers = self.model.get_providers()
            if "CUDAExecutionProvider" not in current_providers:
                warnings.warn(
                    f"Attempt to set CUDAExecutionProvider failed. Current providers: {current_providers}."
                    "If you are using CUDA 12.x, install onnxruntime-gpu via "
                    "`pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/`",
                    RuntimeWarning,
                )

    def load_onnx_model(self) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    def onnx_embed(self, *args: Any, **kwargs: Any) -> OnnxOutputContext:
        raise NotImplementedError("Subclasses must implement this method")


class EmbeddingWorker(Worker, Generic[T]):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs: Any,
    ) -> OnnxModel[T]:
        raise NotImplementedError()

    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs: Any,
    ):
        self.model = self.init_embedding(model_name, cache_dir, **kwargs)

    @classmethod
    def start(cls, model_name: str, cache_dir: str, **kwargs: Any) -> "EmbeddingWorker[T]":
        return cls(model_name=model_name, cache_dir=cache_dir, **kwargs)

    def process(self, items: Iterable[tuple[int, Any]]) -> Iterable[tuple[int, Any]]:
        raise NotImplementedError("Subclasses must implement this method")
