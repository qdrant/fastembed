from typing import Any, Type

from fastembed.common.model_description import DenseModelDescription, ModelSource
from fastembed.image.onnx_embedding import OnnxImageEmbedding, OnnxImageEmbeddingWorker

supported_siglip_models: list[DenseModelDescription] = [
    DenseModelDescription(
        model="google/siglip2-base-patch16-224",
        dim=768,
        description="Image embeddings, Multimodal (text&image), 2025 year",
        license="apache-2.0",
        size_in_GB=0.37,
        sources=ModelSource(hf="onnx-community/siglip2-base-patch16-224-ONNX"),
        model_file="onnx/vision_model.onnx",
    ),
]


class SiglipOnnxImageEmbedding(OnnxImageEmbedding):
    """SigLIP vision tower.

    The exported graph returns both the per-patch `last_hidden_state` and the pooled
    `pooler_output`; only the latter is the image embedding, so it must be selected explicitly.
    """

    ONNX_OUTPUT_NAMES = ["pooler_output"]

    @classmethod
    def _get_worker_class(cls) -> Type["OnnxImageEmbeddingWorker"]:
        return SiglipImageEmbeddingWorker

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        return supported_siglip_models


class SiglipImageEmbeddingWorker(OnnxImageEmbeddingWorker):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs: Any) -> OnnxImageEmbedding:
        return SiglipOnnxImageEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
