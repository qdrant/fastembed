from typing import Any, Dict, Iterable, List, Optional, Sequence, Type, Union
from fastembed.rerank.cross_encoder.text_cross_encoder_base import TextCrossEncoderBase
from fastembed.rerank.cross_encoder.onnx_text_cross_encoder import OnnxTextCrossEncoder
from fastembed.common import OnnxProvider

class TextCrossEncoder(TextCrossEncoderBase):
    CROSS_ENCODER_REGISTRY: List[Type[TextCrossEncoderBase]] = [
        OnnxTextCrossEncoder,
    ]

    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        result = []
        for encoder in cls.CROSS_ENCODER_REGISTRY:
            result.extend(encoder.list_supported_models())
        return result

    def __init__(
        self, 
        model_name: str, 
        cache_dir: str = None, 
        threads: int = None, 
        **kwargs
    ):
        super().__init__(model_name, cache_dir, threads, **kwargs)

        for CROSS_ENCODER_TYPE in self.CROSS_ENCODER_REGISTRY:
            supported_models = CROSS_ENCODER_TYPE.list_supported_models()
            if any(
                model_name.lower() == model["model"].lower()
                for model in supported_models
            ):
                self.model = CROSS_ENCODER_TYPE(
                    model_name, 
                    cache_dir, 
                    threads, 
                    **kwargs
                )
                return

        raise ValueError(
            f"Model {model_name} is not supported in TextCrossEncoder."
            "Please check the supported models using `TextCrossEncoder.list_supported_models()`"
        )

    def rerank(
            self, 
            query: str, 
            documents: Union[str, Iterable[str]], 
            batch_size: int = 64,
            parallel: Optional[int] = None,
            **kwargs
    ) -> Iterable[float]:
        return self.model.rerank(query, documents, **kwargs)
