from typing import Any, Iterable, Type


from fastembed.common.types import NumpyArray
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.utils import normalize
from fastembed.text.onnx_embedding import OnnxTextEmbedding, OnnxTextEmbeddingWorker
from fastembed.common.model_description import DenseModelDescription, ModelSource


supported_builtin_pooling_normalized_models: list[DenseModelDescription] = [
    DenseModelDescription(
        model="google/embeddinggemma-300m",
        dim=768,
        description=(
            "Text embeddings, Unimodal (text), multilingual, 2048 input tokens truncation, "
            "Prefixes for queries/documents: `task: search result | query: {content}` for query, "
            "`title: {title | 'none'} | text: {content}` for documents, 2025 year."
        ),
        license="apache-2.0",
        size_in_GB=1.24,
        sources=ModelSource(
            hf="onnx-community/embeddinggemma-300m-ONNX",
        ),
        model_file="onnx/model.onnx",
        additional_files=["onnx/model.onnx_data"],
    ),
]


class BuiltinPoolingNormalizedEmbedding(OnnxTextEmbedding):
    @classmethod
    def _get_worker_class(cls) -> Type[OnnxTextEmbeddingWorker]:
        return BuiltinPoolingNormalizedEmbeddingWorker

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        """Lists the supported models.

        Returns:
            list[DenseModelDescription]: A list of DenseModelDescription objects containing the model information.
        """
        return supported_builtin_pooling_normalized_models

    def _post_process_onnx_output(
        self, output: OnnxOutputContext, **kwargs: Any
    ) -> Iterable[NumpyArray]:
        return normalize(output.model_output)

    def _run_model(
        self, onnx_input: dict[str, Any], onnx_output_names: list[str] | None = None
    ) -> NumpyArray:
        return self.model.run(onnx_output_names, onnx_input)[1]  # type: ignore[union-attr]


class BuiltinPoolingNormalizedEmbeddingWorker(OnnxTextEmbeddingWorker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs: Any,
    ) -> OnnxTextEmbedding:
        return BuiltinPoolingNormalizedEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
