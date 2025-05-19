from typing import Optional, Sequence, Any, Iterable

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fastembed.common import OnnxProvider
from fastembed.common.model_description import (
    PoolingType,
    DenseModelDescription,
)
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.types import NumpyArray
from fastembed.common.utils import normalize, mean_pooling
from fastembed.text.onnx_embedding import OnnxTextEmbedding


@dataclass(frozen=True)
class PostprocessingConfig:
    pooling: PoolingType
    normalization: bool


class CustomTextEmbedding(OnnxTextEmbedding):
    SUPPORTED_MODELS: list[DenseModelDescription] = []
    POSTPROCESSING_MAPPING: dict[str, PostprocessingConfig] = {}

    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence[OnnxProvider]] = None,
        cuda: bool = False,
        device_ids: Optional[list[int]] = None,
        lazy_load: bool = False,
        device_id: Optional[int] = None,
        specific_model_path: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=threads,
            providers=providers,
            cuda=cuda,
            device_ids=device_ids,
            lazy_load=lazy_load,
            device_id=device_id,
            specific_model_path=specific_model_path,
            **kwargs,
        )
        self._pooling = self.POSTPROCESSING_MAPPING[model_name].pooling
        self._normalization = self.POSTPROCESSING_MAPPING[model_name].normalization

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        return cls.SUPPORTED_MODELS

    def _post_process_onnx_output(
        self, output: OnnxOutputContext, **kwargs: Any
    ) -> Iterable[NumpyArray]:
        return self._normalize(self._pool(output.model_output, output.attention_mask))

    def _pool(
        self, embeddings: NumpyArray, attention_mask: Optional[NDArray[np.int64]] = None
    ) -> NumpyArray:
        if self._pooling == PoolingType.CLS:
            return embeddings[:, 0]

        if self._pooling == PoolingType.MEAN:
            if attention_mask is None:
                raise ValueError("attention_mask must be provided for mean pooling")
            return mean_pooling(embeddings, attention_mask)

        if self._pooling == PoolingType.DISABLED:
            return embeddings

        raise ValueError(
            f"Unsupported pooling type {self._pooling}. "
            f"Supported types are: {PoolingType.CLS}, {PoolingType.MEAN}, {PoolingType.DISABLED}."
        )

    def _normalize(self, embeddings: NumpyArray) -> NumpyArray:
        return normalize(embeddings) if self._normalization else embeddings

    @classmethod
    def add_model(
        cls,
        model_description: DenseModelDescription,
        pooling: PoolingType,
        normalization: bool,
    ) -> None:
        cls.SUPPORTED_MODELS.append(model_description)
        cls.POSTPROCESSING_MAPPING[model_description.model] = PostprocessingConfig(
            pooling=pooling, normalization=normalization
        )
