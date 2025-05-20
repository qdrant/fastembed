from dataclasses import asdict
from typing import Union, Iterable, Optional, Any, Type

from fastembed.common.model_description import DenseModelDescription, ModelSource
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.types import NumpyArray
from fastembed.late_interaction.late_interaction_embedding_base import (
    LateInteractionTextEmbeddingBase,
)
from fastembed.text.onnx_embedding import OnnxTextEmbedding
from fastembed.text.onnx_text_model import TextEmbeddingWorker


supported_token_embeddings_models = [
    DenseModelDescription(
        model="jinaai/jina-embeddings-v2-small-en-tokens",
        dim=512,
        description="Text embeddings, Unimodal (text), English, 8192 input tokens truncation,"
        " Prefixes for queries/documents: not necessary, 2023 year.",
        license="apache-2.0",
        size_in_GB=0.12,
        sources=ModelSource(hf="xenova/jina-embeddings-v2-small-en"),
        model_file="onnx/model.onnx",
    ),
]


class TokenEmbeddingsModel(OnnxTextEmbedding, LateInteractionTextEmbeddingBase):
    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        """Lists the supported models.

        Returns:
            list[DenseModelDescription]: A list of DenseModelDescription objects containing the model information.
        """
        return supported_token_embeddings_models

    @classmethod
    def list_supported_models(cls) -> list[dict[str, Any]]:
        """Lists the supported models.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing the model information.
        """
        return [asdict(model) for model in cls._list_supported_models()]

    @classmethod
    def _get_worker_class(cls) -> Type[TextEmbeddingWorker[NumpyArray]]:
        return TokensEmbeddingWorker

    def _post_process_onnx_output(
        self, output: OnnxOutputContext, **kwargs: Any
    ) -> Iterable[NumpyArray]:
        # Size: (batch_size, sequence_length, hidden_size)
        embeddings = output.model_output
        # Size: (batch_size, sequence_length)
        assert output.attention_mask is not None
        masks = output.attention_mask

        # For each document we only select those embeddings that are not masked out
        for i in range(embeddings.shape[0]):
            yield embeddings[i, masks[i] == 1]

    def embed(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        yield from super().embed(documents, batch_size=batch_size, parallel=parallel, **kwargs)


class TokensEmbeddingWorker(TextEmbeddingWorker[NumpyArray]):
    def init_embedding(
        self, model_name: str, cache_dir: str, **kwargs: Any
    ) -> TokenEmbeddingsModel:
        return TokenEmbeddingsModel(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
