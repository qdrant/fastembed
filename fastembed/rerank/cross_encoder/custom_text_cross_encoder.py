from typing import Sequence, Any, Type

from fastembed.common import OnnxProvider
from fastembed.common.model_description import BaseModelDescription
from fastembed.common.types import Device
from fastembed.rerank.cross_encoder.onnx_text_cross_encoder import OnnxTextCrossEncoder
from fastembed.rerank.cross_encoder.onnx_text_model import TextRerankerWorker


class CustomTextCrossEncoder(OnnxTextCrossEncoder):
    SUPPORTED_MODELS: list[BaseModelDescription] = []

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

    @classmethod
    def _list_supported_models(cls) -> list[BaseModelDescription]:
        return cls.SUPPORTED_MODELS

    @classmethod
    def _get_worker_class(cls) -> Type[TextRerankerWorker]:
        return CustomTextCrossEncoderWorker

    def _get_worker_init_kwargs(self) -> dict[str, Any]:
        return {"model_description": self.model_description}

    @classmethod
    def add_model(
        cls,
        model_description: BaseModelDescription,
    ) -> None:
        cls.SUPPORTED_MODELS.append(model_description)


class CustomTextCrossEncoderWorker(TextRerankerWorker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
        model_description: BaseModelDescription | None = None,
        **kwargs: Any,
    ) -> CustomTextCrossEncoder:
        if model_description is None:
            raise ValueError(
                "`model_description` is required to initialize a custom model in a worker "
                "process, it is provided by `CustomTextCrossEncoder._get_worker_init_kwargs`"
            )
        # custom models live in a class-level registry, which spawned workers don't inherit
        CustomTextCrossEncoder.add_model(model_description)
        return CustomTextCrossEncoder(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
