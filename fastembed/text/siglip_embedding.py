from typing import Any, Type

from fastembed.common.model_description import DenseModelDescription, ModelSource
from fastembed.text.onnx_embedding import OnnxTextEmbedding, OnnxTextEmbeddingWorker

supported_siglip_models: list[DenseModelDescription] = [
    DenseModelDescription(
        model="google/siglip2-base-patch16-224",
        dim=768,
        description=(
            "Text embeddings, Multimodal (text&image), multilingual, 64 input tokens truncation, "
            "Prefixes for queries/documents: not necessary, 2025 year"
        ),
        license="apache-2.0",
        size_in_GB=1.13,
        sources=ModelSource(hf="onnx-community/siglip2-base-patch16-224-ONNX"),
        model_file="onnx/text_model.onnx",
    ),
]


class SiglipOnnxTextEmbedding(OnnxTextEmbedding):
    """SigLIP text tower.

    SigLIP always pools the hidden state at the last sequence position, whether or not it is
    padding, so every batch must be padded to the model's fixed training length (rather than to
    the longest sequence in the batch) or the resulting embeddings become dependent on what else
    is in the batch.

    The exported graph also returns both `last_hidden_state` and `pooler_output`; only the
    latter is the text embedding, so it must be selected explicitly.
    """

    ONNX_OUTPUT_NAMES = ["pooler_output"]

    @classmethod
    def _get_worker_class(cls) -> Type[OnnxTextEmbeddingWorker]:
        return SiglipTextEmbeddingWorker

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        return supported_siglip_models

    def load_onnx_model(self) -> None:
        super().load_onnx_model()
        if self.tokenizer is not None:
            truncation = self.tokenizer.truncation
            padding = self.tokenizer.padding
            if truncation and padding and padding.get("length") is None:
                self.tokenizer.enable_padding(
                    direction=padding["direction"],
                    pad_id=padding["pad_id"],
                    pad_type_id=padding["pad_type_id"],
                    pad_token=padding["pad_token"],
                    length=truncation["max_length"],
                )


class SiglipTextEmbeddingWorker(OnnxTextEmbeddingWorker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs: Any,
    ) -> OnnxTextEmbedding:
        return SiglipOnnxTextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
