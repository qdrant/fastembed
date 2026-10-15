from typing import Any, Iterable, Type


from fastembed.common.types import NumpyArray
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.utils import normalize
from fastembed.image.onnx_embedding import OnnxImageEmbedding
from fastembed.image.onnx_image_model import ImageEmbeddingWorker
from fastembed.common.model_description import DenseModelDescription, ModelSource

supported_normalized_models: list[DenseModelDescription] = [
    DenseModelDescription(
        model="nomic-ai/nomic-embed-vision-v1.5",
        dim=768,
        description="Image embeddings, Multimodal (text&image), 2024 year",
        license="apache-2.0",
        size_in_GB=0.37,
        sources=ModelSource(hf="nomic-ai/nomic-embed-vision-v1.5"),
        model_file="onnx/model.onnx",
    ),
    DenseModelDescription(
        model="nomic-ai/nomic-embed-vision-v1.5-Q",
        dim=768,
        description="Image embeddings, Multimodal (text&image), 2024 year",
        license="apache-2.0",
        size_in_GB=0.1,
        sources=ModelSource(hf="nomic-ai/nomic-embed-vision-v1.5"),
        model_file="onnx/model_quantized.onnx",
    ),
]


class NormalizedEmbedding(OnnxImageEmbedding):
    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        """
        Lists the supported models.

        Returns:
            list[DenseModelDescription]: A list of DenseModelDescription objects containing the model information.
        """
        return supported_normalized_models

    @classmethod
    def _get_worker_class(cls) -> Type["ImageEmbeddingWorker[NumpyArray]"]:
        return NormalizedEmbeddingWorker

    def _post_process_onnx_output(
        self, output: OnnxOutputContext, **kwargs: Any
    ) -> Iterable[NumpyArray]:
        # The model emits last_hidden_state, which onnx_embed flattens to (batch, tokens * dim).
        # Recover the token axis, take the CLS token (index 0) and normalize, matching the reference
        # F.normalize(last_hidden_state[:, 0], p=2, dim=1).
        dim = self.model_description.dim
        assert dim is not None, "Model description is missing the embedding dim"
        hidden_states = output.model_output.reshape(output.model_output.shape[0], -1, dim)
        return normalize(hidden_states[:, 0])


class NormalizedEmbeddingWorker(ImageEmbeddingWorker[NumpyArray]):
    def init_embedding(
        self, model_name: str, cache_dir: str, **kwargs: Any
    ) -> NormalizedEmbedding:
        return NormalizedEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
