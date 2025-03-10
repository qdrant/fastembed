from typing import Optional, Sequence, Any

from fastembed.common import OnnxProvider
from fastembed.common.model_description import (
    DenseModelDescription,
)
from fastembed.rerank.cross_encoder.onnx_text_cross_encoder import OnnxTextCrossEncoder


class CustomCrossEncoderModel(OnnxTextCrossEncoder):
    SUPPORTED_MODELS: list[DenseModelDescription] = []

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

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        return cls.SUPPORTED_MODELS

    @classmethod
    def add_model(
        cls,
        model_description: DenseModelDescription,
    ) -> None:
        cls.SUPPORTED_MODELS.append(model_description)
