from typing import Any, Dict, Iterable, List, Optional, Sequence, Type, Union

from fastembed.rerank.cross_encoder.text_cross_encoder_base import TextCrossEncoderBase
from fastembed.rerank.cross_encoder.onnx_text_cross_encoder import OnnxTextCrossEncoder
from fastembed.rerank.cross_encoder.baai_text_cross_encoder import BAAITextCrossEncoder
from fastembed.common import OnnxProvider


class TextCrossEncoder(TextCrossEncoderBase):
    CROSS_ENCODER_REGISTRY: List[Type[TextCrossEncoderBase]] = [
        OnnxTextCrossEncoder,
        BAAITextCrossEncoder,
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
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence[OnnxProvider]] = None,
        **kwargs,
    ):
        super().__init__(model_name, cache_dir, threads, **kwargs)

        for CROSS_ENCODER_TYPE in self.CROSS_ENCODER_REGISTRY:
            supported_models = CROSS_ENCODER_TYPE.list_supported_models()
            if any(model_name.lower() == model["model"].lower() for model in supported_models):
                self.model = CROSS_ENCODER_TYPE(
                    model_name=model_name,
                    cache_dir=cache_dir,
                    threads=threads,
                    providers=providers,
                    **kwargs,
                )
                return

        raise ValueError(
            f"Model {model_name} is not supported in TextCrossEncoder."
            "Please check the supported models using `TextCrossEncoder.list_supported_models()`"
        )

    def rerank(
        self, query: str, documents: Union[str, Iterable[str]], batch_size: int = 64, **kwargs
    ) -> Iterable[float]:
        return self.model.rerank(query, documents, batch_size=batch_size, **kwargs)
