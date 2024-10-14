from typing import Any, Dict, Iterable, List, Optional, Sequence, Type

from fastembed.rerank.cross_encoder.text_cross_encoder_base import TextCrossEncoderBase
from fastembed.rerank.cross_encoder.onnx_text_cross_encoder import OnnxTextCrossEncoder
from fastembed.common import OnnxProvider


class TextCrossEncoder(TextCrossEncoderBase):
    CROSS_ENCODER_REGISTRY: List[Type[TextCrossEncoderBase]] = [
        OnnxTextCrossEncoder,
    ]

    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        """Lists the supported models.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the model information.

            Example:
                ```
                [
                    {
                        "model": "Xenova/ms-marco-MiniLM-L-6-v2",
                        "size_in_GB": 0.08,
                        "sources": {
                            "hf": "Xenova/ms-marco-MiniLM-L-6-v2",
                        },
                        "model_file": "onnx/model.onnx",
                        "description": "MiniLM-L-6-v2 model optimized for re-ranking tasks.",
                        "license": "apache-2.0",
                    }
                ]
                ```
        """
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
        cuda: bool = False,
        device_ids: Optional[List[int]] = None,
        lazy_load: bool = False,
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
                    cuda=cuda,
                    device_ids=device_ids,
                    lazy_load=lazy_load,
                    **kwargs,
                )
                return

        raise ValueError(
            f"Model {model_name} is not supported in TextCrossEncoder."
            "Please check the supported models using `TextCrossEncoder.list_supported_models()`"
        )

    def rerank(
        self, query: str, documents: Iterable[str], batch_size: int = 64, **kwargs
    ) -> Iterable[float]:
        """Rerank a list of documents based on a query.

        Args:
            query: Query to rerank the documents against
            documents: Iterator of documents to rerank
            batch_size: Batch size for reranking

        Returns:
            Iterable of scores for each document
        """
        yield from self.model.rerank(query, documents, batch_size=batch_size, **kwargs)
