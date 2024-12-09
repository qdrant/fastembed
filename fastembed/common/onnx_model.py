import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Self
)

import numpy as np
import onnxruntime as ort

from fastembed.common.types import OnnxProvider # type: ignore
from fastembed.parallel_processor import Worker

# Holds type of the embedding result
T = TypeVar("T")


@dataclass
class OnnxOutputContext:
    model_output: np.ndarray
    attention_mask: Optional[np.ndarray] = None
    input_ids: Optional[np.ndarray] = None


class OnnxModel(Generic[T]):
    @classmethod
    def _get_worker_class(cls) -> Type["EmbeddingWorker"]:
        raise NotImplementedError("Subclasses must implement this method")

    def _post_process_onnx_output(self, output: OnnxOutputContext) -> Iterable[T]:
        raise NotImplementedError("Subclasses must implement this method")

    def __init__(self) -> None:
        self.model: ort.InferenceSession = None
        self.tokenizer = None

    def _preprocess_onnx_input(
        self, onnx_input: Dict[str, np.ndarray], **kwargs: Any
    ) -> Dict[str, np.ndarray]:
        """
        Preprocess the onnx input.
        """
        return onnx_input

    def _load_onnx_model(
        self: Self,
        model_dir: Path,
        model_file: str,
        threads: Optional[int],
        providers: Optional[Sequence[OnnxProvider]] = None,
        cuda: bool = False,
        device_id: Optional[int] = None,
    ) -> None:
        model_path = model_dir / model_file
        # List of Execution Providers: https://onnxruntime.ai/docs/execution-providers

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
        requested_provider_names = []
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

    def onnx_embed(self: Self, *args: Any, **kwargs: Any) -> OnnxOutputContext:
        raise NotImplementedError("Subclasses must implement this method")


class EmbeddingWorker(Worker):
    def init_embedding(
        self: Self,
        model_name: str,
        cache_dir: str,
        **kwargs: Any,
    ) -> OnnxModel:
        raise NotImplementedError()

    def __init__(
        self: Self,
        model_name: str,
        cache_dir: str,
        **kwargs: Any,
    ):
        self.model = self.init_embedding(model_name, cache_dir, **kwargs)

    @classmethod
    def start(cls: type[Self], **kwargs: Any) -> "EmbeddingWorker":
        model_name = kwargs.get("model_name", None)
        if model_name is None:
            raise ValueError("model_name must be provided")
        cache_dir = kwargs.get("cache_dir", None)
        if cache_dir is None:
            raise ValueError("cache_dir must be provided")
        return cls(model_name=model_name, cache_dir=cache_dir, **kwargs)

    def process(self: Self, items: Iterable[Tuple[int, Any]]) -> Iterable[Tuple[int, Any]]:
        raise NotImplementedError("Subclasses must implement this method")
