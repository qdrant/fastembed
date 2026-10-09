from typing import Sequence, Any, Iterable, Type
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fastembed.common import OnnxProvider
from fastembed.common.model_description import (
    PoolingType,
    DenseModelDescription,
)
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.types import NumpyArray, Device
from fastembed.common.utils import normalize, mean_pooling, last_token_pooling
from fastembed.text.onnx_embedding import OnnxTextEmbedding
from fastembed.text.onnx_text_model import TextEmbeddingWorker


@dataclass(frozen=True)
class PostprocessingConfig:
    pooling: PoolingType
    normalization: bool
    output_name: str | None = None


class CustomTextEmbedding(OnnxTextEmbedding):
    SUPPORTED_MODELS: list[DenseModelDescription] = []
    POSTPROCESSING_MAPPING: dict[str, PostprocessingConfig] = {}

    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        threads: int | None = None,
        providers: Sequence[OnnxProvider] | None = None,
        cuda: bool | Device = Device.AUTO,
        device_ids: list[int] | None = None,
        lazy_load: bool = False,
        device_id: int | None = None,
        specific_model_path: str | None = None,
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
        postprocessing_config = self.POSTPROCESSING_MAPPING[self.model_description.model]
        self._pooling = postprocessing_config.pooling
        self._normalization = postprocessing_config.normalization
        if postprocessing_config.output_name is not None:
            self.ONNX_OUTPUT_NAMES = [postprocessing_config.output_name]

    def load_onnx_model(self) -> None:
        super().load_onnx_model()
        # with eager loading this runs inside super().__init__(), before ONNX_OUTPUT_NAMES is set,
        # so the output name is taken from the registered config
        output_name = self.POSTPROCESSING_MAPPING[self.model_description.model].output_name
        output_names = [output.name for output in self.model.get_outputs()]  # type: ignore[union-attr]
        if output_name is not None and output_name not in output_names:
            raise ValueError(
                f"Output {output_name!r} not found in the model, available outputs: {output_names}"
            )

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        return cls.SUPPORTED_MODELS

    @classmethod
    def _get_worker_class(cls) -> Type["TextEmbeddingWorker[NumpyArray]"]:
        return CustomTextEmbeddingWorker

    def _get_worker_init_kwargs(self) -> dict[str, Any]:
        return {
            "model_description": self.model_description,
            "postprocessing_config": self.POSTPROCESSING_MAPPING[self.model_description.model],
        }

    def _post_process_onnx_output(
        self, output: OnnxOutputContext, **kwargs: Any
    ) -> Iterable[NumpyArray]:
        embeddings = self._normalize(self._pool(output.model_output, output.attention_mask))
        # mean pooling returns float64, float embeddings are cast back to the dtype of the model
        # after normalization, integer outputs are kept as is, since the cast would truncate them
        if np.issubdtype(output.model_output.dtype, np.floating):
            return embeddings.astype(output.model_output.dtype, copy=False)
        return embeddings

    def _pool(
        self, embeddings: NumpyArray, attention_mask: NDArray[np.int64] | None = None
    ) -> NumpyArray:
        if self._pooling == PoolingType.CLS:
            return embeddings[:, 0]

        if self._pooling == PoolingType.MEAN:
            if attention_mask is None:
                raise ValueError("attention_mask must be provided for mean pooling")
            return mean_pooling(embeddings, attention_mask)

        if self._pooling == PoolingType.LAST_TOKEN:
            if attention_mask is None:
                raise ValueError("attention_mask must be provided for last token pooling")
            return last_token_pooling(embeddings, attention_mask)

        if self._pooling == PoolingType.DISABLED:
            if embeddings.ndim != 2:
                raise ValueError(
                    f"{PoolingType.DISABLED} pooling expects the model to output sentence "
                    f"embeddings of shape (batch_size, dim), got an output of shape "
                    f"{embeddings.shape}. Use {PoolingType.CLS}, {PoolingType.MEAN} or "
                    f"{PoolingType.LAST_TOKEN} pooling, or set `output_name` to a pooled output "
                    "of the model, e.g. `sentence_embedding`."
                )
            return embeddings

        raise ValueError(
            f"Unsupported pooling type {self._pooling}. "
            f"Supported types are: {PoolingType.CLS}, {PoolingType.MEAN}, "
            f"{PoolingType.LAST_TOKEN}, {PoolingType.DISABLED}."
        )

    def _normalize(self, embeddings: NumpyArray) -> NumpyArray:
        return normalize(embeddings) if self._normalization else embeddings

    @classmethod
    def add_model(
        cls,
        model_description: DenseModelDescription,
        pooling: PoolingType,
        normalization: bool,
        output_name: str | None = None,
    ) -> None:
        cls.SUPPORTED_MODELS.append(model_description)
        cls.POSTPROCESSING_MAPPING[model_description.model] = PostprocessingConfig(
            pooling=pooling, normalization=normalization, output_name=output_name
        )


class CustomTextEmbeddingWorker(TextEmbeddingWorker[NumpyArray]):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
        model_description: DenseModelDescription | None = None,
        postprocessing_config: PostprocessingConfig | None = None,
        **kwargs: Any,
    ) -> CustomTextEmbedding:
        if model_description is None or postprocessing_config is None:
            raise ValueError(
                "`model_description` and `postprocessing_config` are required to initialize a "
                "custom model in a worker process, they are provided by "
                "`CustomTextEmbedding._get_worker_init_kwargs`"
            )
        # custom models live in a class-level registry, which spawned workers don't inherit
        CustomTextEmbedding.add_model(
            model_description,
            pooling=postprocessing_config.pooling,
            normalization=postprocessing_config.normalization,
            output_name=postprocessing_config.output_name,
        )
        return CustomTextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
