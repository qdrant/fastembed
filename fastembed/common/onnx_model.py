import importlib.util
import os
import json
import platform
import sys
import warnings
from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path
from typing import Any, Generic, Iterable, Sequence, Type, TypeVar

import numpy as np
import onnxruntime as ort
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument

from loguru import logger
from numpy.typing import NDArray
from tokenizers import Tokenizer

from fastembed.common.onnx_external_data import ONNX_SNAPSHOTS_DIR, link_external_data
from fastembed.common.types import OnnxProvider, NumpyArray, Device
from fastembed.common.utils import MAX_BATCH_TOKENS
from fastembed.parallel_processor import Worker

# onnxruntime imported before fastembed: stop telemetry for the sessions created here.
if os.environ.get("ORT_DISABLE_TELEMETRY", "1").lower() not in ("0", "false") and hasattr(
    ort, "disable_telemetry_events"
):
    ort.disable_telemetry_events()

# Holds type of the embedding result
T = TypeVar("T")

_SHRINKAGE_KEY = "memory.enable_memory_arena_shrinkage"


@cache
def preload_pip_cuda_libraries() -> None:
    """Load the CUDA and cuDNN libraries of the `nvidia-*` pip packages before CUDA sessions.

    onnxruntime uses them only after `onnxruntime.preload_dlls()`; otherwise it looks for the
    system's libraries. The call is skipped where it would only print: without the packages, or
    when torch has already loaded the libraries.
    """
    if (
        not hasattr(ort, "preload_dlls")
        or importlib.util.find_spec("nvidia") is None
        or torch_loaded_cuda_libraries()
    ):
        return
    ort.preload_dlls()


def ort_cuda_version() -> str | None:
    """The CUDA version onnxruntime is built for, such as "12.2"; None for builds without CUDA."""
    try:
        from onnxruntime.capi import build_and_package_info
    except ImportError:
        return None
    return getattr(build_and_package_info, "cuda_version", None)


def torch_loaded_cuda_libraries() -> bool:
    """Whether an imported torch built for the same major CUDA version loaded the libraries.

    On Windows, `onnxruntime.preload_dlls()` then only prints that it skips them.
    """
    torch = sys.modules.get("torch")
    if platform.system() != "Windows" or torch is None:
        return False
    torch_cuda = getattr(getattr(torch, "version", None), "cuda", None)
    ort_cuda = ort_cuda_version()
    if not torch_cuda or not ort_cuda:
        return False
    return torch_cuda.split(".")[0] == ort_cuda.split(".")[0]


def cuda_fallback_message(current_providers: list[str]) -> str:
    """Why a CUDA session fell back to other providers, and how to install matching libraries."""
    cuda_version = ort_cuda_version()
    cuda = f"CUDA {cuda_version.split('.')[0]}" if cuda_version else "a matching CUDA version"
    return (
        f"Attempt to set CUDAExecutionProvider failed. Current providers: {current_providers}. "
        f"onnxruntime-gpu {ort.__version__} needs {cuda} and cuDNN 9. To install both from pip: "
        '`pip install -U "onnxruntime-gpu[cuda,cudnn]"` (CUDA 13, NVIDIA driver 580 or newer), '
        'or `pip install "onnxruntime-gpu[cuda,cudnn]>=1.21,<1.27"` (CUDA 12).'
    )


@dataclass
class OnnxOutputContext:
    model_output: NumpyArray
    attention_mask: NDArray[np.int64] | None = None
    input_ids: NDArray[np.int64] | None = None
    metadata: dict[str, Any] | None = None


class OnnxModel(Generic[T]):
    EXPOSED_SESSION_OPTIONS = ("enable_cpu_mem_arena",)
    # set by the concrete model classes in their __init__
    _model_dir: Path
    # the `max_batch_tokens` model option, see `_batch_token_budget`
    _max_batch_tokens: int | None = None
    run_options: ort.RunOptions | None = None

    @classmethod
    def _get_worker_class(cls) -> Type["EmbeddingWorker[T]"]:
        raise NotImplementedError("Subclasses must implement this method")

    def _get_worker_init_kwargs(self) -> dict[str, Any]:
        """Additional kwargs a worker process needs to reconstruct this model.

        Workers are started with `spawn`/`forkserver`, hence they don't inherit class-level state
        which has been set up in runtime, e.g. models registered via `add_custom_model`.
        Such state has to be shipped to the workers explicitly.

        Returns:
            dict[str, Any]: kwargs to pass to `_get_worker_class().init_embedding`.
        """
        return {}

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
        self.model: ort.InferenceSession | None = None
        self.tokenizer: Tokenizer | None = None

    def _preprocess_onnx_input(
        self, onnx_input: dict[str, NumpyArray], **kwargs: Any
    ) -> dict[str, NumpyArray]:
        """
        Preprocess the onnx input.
        """
        return onnx_input

    def _batch_token_budget(self) -> int | None:
        """Padded tokens per batch, None to keep batches of `batch_size`.

        `MAX_BATCH_TOKENS` on every provider unless the `max_batch_tokens` model option sets
        another number, 0 turns the budget off."""
        if self._max_batch_tokens is not None:
            return self._max_batch_tokens or None
        return MAX_BATCH_TOKENS

    @cached_property
    def _attention_heads(self) -> int | None:
        """Attention heads per layer from config.json, None when the model doesn't say."""
        config_path = getattr(self, "_model_dir", Path()) / "config.json"
        if not config_path.is_file():
            return None
        with open(config_path, encoding="utf-8") as config_file:
            config = json.load(config_file)
        heads = config.get("num_attention_heads", config.get("n_head"))
        return heads if type(heads) is int and heads > 0 else None

    def _load_onnx_model(
        self,
        model_dir: Path,
        model_file: str,
        threads: int | None,
        providers: Sequence[OnnxProvider] | None = None,
        cuda: bool | Device = Device.AUTO,
        device_id: int | None = None,
        extra_session_options: dict[str, Any] | None = None,
        additional_files: list[str] | None = None,
    ) -> None:
        # List of Execution Providers: https://onnxruntime.ai/docs/execution-providers
        available_providers = ort.get_available_providers()
        cuda_available = "CUDAExecutionProvider" in available_providers
        explicit_cuda = cuda is True or cuda == Device.CUDA

        if explicit_cuda and providers is not None:
            warnings.warn(
                f"`cuda` and `providers` are mutually exclusive parameters, "
                f"cuda: {cuda}, providers: {providers}. If you'd like to use providers, cuda should be one of "
                f"[False, Device.CPU, Device.AUTO].",
                category=UserWarning,
                stacklevel=6,
            )

        if providers is not None:
            onnx_providers = list(providers)
        elif explicit_cuda or (cuda == Device.AUTO and cuda_available):
            if device_id is None:
                onnx_providers = ["CUDAExecutionProvider"]
            else:
                onnx_providers = [("CUDAExecutionProvider", {"device_id": device_id})]
        else:
            onnx_providers = ["CPUExecutionProvider"]

        requested_provider_names: list[str] = []
        for provider in onnx_providers:
            # check providers available
            provider_name = provider if isinstance(provider, str) else provider[0]
            requested_provider_names.append(provider_name)
            if provider_name not in available_providers:
                raise ValueError(
                    f"Provider {provider_name} is not available. Available providers: {available_providers}"
                )

        if "CUDAExecutionProvider" in requested_provider_names:
            preload_pip_cuda_libraries()

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # onnxruntime can reuse the memory pattern of a larger input shape, and then shrinkage can fail to release that memory.
        so.enable_mem_pattern = False

        if threads is not None:
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = threads

        if extra_session_options is not None:
            self.add_extra_session_options(so, extra_session_options)

        model_path = model_dir / model_file
        link_error: OSError | None = None
        try:
            model_path = link_external_data(model_dir, model_file, additional_files or [])
        except OSError as e:
            # e.g. a read-only cache, which onnxruntime<1.24 loads from all the same
            link_error = e

        try:
            self.model = ort.InferenceSession(
                str(model_path), providers=onnx_providers, sess_options=so
            )
        except Exception:
            if link_error is not None:
                logger.warning(
                    f"Could not link the files of {model_path} into {ONNX_SNAPSHOTS_DIR}: "
                    f"{link_error}. onnxruntime>=1.24 refuses external data that resolves "
                    "outside of the model directory, as in a huggingface_hub>=1.32 cache. "
                    "Download the model with fastembed into a writable cache_dir on a "
                    "filesystem with hardlinks, or delete it from the cache and download it "
                    "again with HF_HUB_DISABLE_SHARED_BLOBS=1."
                )
            raise
        self.run_options = self._shrink_run_options(self._arenas_to_shrink(so))
        if "CUDAExecutionProvider" in requested_provider_names:
            assert self.model is not None
            current_providers = self.model.get_providers()
            if "CUDAExecutionProvider" not in current_providers:
                warnings.warn(cuda_fallback_message(current_providers), RuntimeWarning)

    def _arenas_to_shrink(self, so: ort.SessionOptions) -> list[str]:
        # onnxruntime rejects "cpu:0" in the list when the CPU arena is off.
        arenas = ["cpu:0"] if so.enable_cpu_mem_arena else []
        cuda_options = self.model.get_provider_options().get("CUDAExecutionProvider")  # type: ignore[union-attr]
        if cuda_options is None:
            return arenas
        # An external CUDA allocator replaces the CUDA arena, so there is nothing to shrink.
        external_allocator = all(
            cuda_options.get(name, "0") != "0"
            for name in ("gpu_external_alloc", "gpu_external_free")
        )
        # Shrinkage could free memory that a captured CUDA graph uses again in later runs.
        cuda_graph = cuda_options.get("enable_cuda_graph", "0") != "0"
        if not external_allocator and not cuda_graph:
            arenas.append(f"gpu:{cuda_options['device_id']}")
        return arenas

    @staticmethod
    def _shrink_run_options(arenas: list[str]) -> ort.RunOptions | None:
        if not arenas:
            return None
        run_options = ort.RunOptions()
        run_options.add_run_config_entry(_SHRINKAGE_KEY, ";".join(arenas))
        return run_options

    def _run_model_with_shrinkage(
        self, output_names: list[str] | None, onnx_input: dict[str, Any]
    ) -> list[Any]:
        run_options = self.run_options
        try:
            return self.model.run(output_names, onnx_input, run_options)  # type: ignore[union-attr]
        except InvalidArgument:
            if run_options is None:
                raise
            # Builds with jemalloc, mimalloc, AddressSanitizer, or 32 bits have no CPU arena and
            # reject "cpu:0", and Python can't detect them, so retry without "cpu:0". The GPU
            # entries come from fastembed's own checks; if onnxruntime rejects them as well,
            # retry without the list.
            arenas = run_options.get_run_config_entry(_SHRINKAGE_KEY).split(";")
            accepted = [arena for arena in arenas if arena != "cpu:0"]
            if accepted == arenas:
                accepted = []
            fallback = self._shrink_run_options(accepted)
            try:
                outputs = self.model.run(output_names, onnx_input, fallback)  # type: ignore[union-attr]
            except InvalidArgument:
                if fallback is None:
                    raise
                accepted, fallback = [], None
                outputs = self.model.run(output_names, onnx_input)  # type: ignore[union-attr]
            logger.warning(
                f"onnxruntime rejected the memory arena shrink list {';'.join(arenas)}, so "
                "fastembed releases arena memory after each run only for "
                f"{';'.join(accepted) or 'no arena'}."
            )
            self.run_options = fallback
            return outputs

    @classmethod
    def _select_exposed_session_options(cls, model_kwargs: dict[str, Any]) -> dict[str, Any]:
        """A convenience method to select the exposed session options in models

        Args:
            model_kwargs (dict[str, Any]): The model kwargs.

        Returns:
            dict[str, Any]: a dict with filtered exposed session options.
        """
        return {k: v for k, v in model_kwargs.items() if k in cls.EXPOSED_SESSION_OPTIONS}

    @staticmethod
    def _select_max_batch_tokens(model_kwargs: dict[str, Any]) -> int | None:
        """The `max_batch_tokens` model option, None when it isn't set.

        Raises:
            ValueError: If it is set to anything but an int >= 0.
        """
        max_batch_tokens = model_kwargs.get("max_batch_tokens")
        if max_batch_tokens is not None and (
            isinstance(max_batch_tokens, bool)
            or not isinstance(max_batch_tokens, int)
            or max_batch_tokens < 0
        ):
            raise ValueError(f"max_batch_tokens must be an int >= 0, got {max_batch_tokens!r}")
        return max_batch_tokens

    @classmethod
    def add_extra_session_options(
        cls, session_options: ort.SessionOptions, extra_options: dict[str, Any]
    ) -> None:
        """Add extra session options to the existing options object in-place

        Args:
            session_options (ort.SessionOptions): The existing session options object.
            extra_options (dict[str, Any]): The extra session options available in cls.EXPOSED_SESSION_OPTIONS.

        Returns:
            None
        """
        for option in extra_options:
            assert (
                option in cls.EXPOSED_SESSION_OPTIONS
            ), f"{option} is unknown or not exposed (exposed options: {cls.EXPOSED_SESSION_OPTIONS})"
        if "enable_cpu_mem_arena" in extra_options:
            session_options.enable_cpu_mem_arena = extra_options["enable_cpu_mem_arena"]

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
