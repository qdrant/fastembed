from pathlib import Path
from typing import Any, Iterable, Type

import numpy as np

from fastembed.common.types import NumpyArray
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.preprocessor_utils import load_tokenizer
from fastembed.text.onnx_embedding import OnnxTextEmbedding, OnnxTextEmbeddingWorker
from fastembed.common.model_description import DenseModelDescription, ModelSource

# Context length for exports whose tokenizer_config.json carries transformers' "unknown"
# sentinel instead of a usable `model_max_length`.
DEFAULT_MAX_LENGTHS: dict[str, int] = {
    "google/embeddinggemma-2": 8192,
    "google/embeddinggemma-2-q": 8192,
}


supported_builtin_sentence_embedding_models: list[DenseModelDescription] = [
    DenseModelDescription(
        model="google/embeddinggemma-300m",
        dim=768,
        description=(
            "Text embeddings, Unimodal (text), multilingual, 2048 input tokens truncation, "
            "Prefixes for queries/documents: `task: search result | query: {content}` for query, "
            "`title: {title | 'none'} | text: {content}` for documents, 2025 year."
        ),
        license="gemma",
        size_in_GB=1.24,
        sources=ModelSource(
            hf="onnx-community/embeddinggemma-300m-ONNX",
        ),
        model_file="onnx/model.onnx",
        additional_files=["onnx/model.onnx_data"],
    ),
    DenseModelDescription(
        model="google/embeddinggemma-2",
        dim=768,
        description=(
            "Text embeddings, Multimodal model used text-only, multilingual, 8192 input tokens "
            "truncation, Prefixes for queries/documents: `task: search result | query: {content}` "
            "for query, `title: {title | 'none'} | text: {content}` for documents, 2026 year."
        ),
        license="apache-2.0",
        size_in_GB=1.08,
        sources=ModelSource(
            hf="onnx-community/embeddinggemma-2-ONNX",
        ),
        model_file="onnx/model.onnx",
        additional_files=["onnx/model.onnx_data"],
    ),
    DenseModelDescription(
        model="google/embeddinggemma-2-Q",
        dim=768,
        description=(
            "Text embeddings, Multimodal model used text-only, multilingual, 8192 input tokens "
            "truncation, Prefixes for queries/documents: `task: search result | query: {content}` "
            "for query, `title: {title | 'none'} | text: {content}` for documents, int8 weights, "
            "2026 year."
        ),
        license="apache-2.0",
        size_in_GB=0.31,
        sources=ModelSource(
            hf="onnx-community/embeddinggemma-2-ONNX",
        ),
        model_file="onnx/model_quantized.onnx",
        additional_files=["onnx/model_quantized.onnx_data"],
    ),
    DenseModelDescription(
        model="ibm-granite/granite-embedding-small-english-r2",
        dim=384,
        description=(
            "Text embeddings, Unimodal (text), English, 8192 input tokens truncation, "
            "Prefixes for queries/documents: not necessary, embeddings are not normalized, 2025 year."
        ),
        license="apache-2.0",
        size_in_GB=0.18,
        sources=ModelSource(
            hf="onnx-community/granite-embedding-small-english-r2-ONNX",
        ),
        model_file="onnx/model.onnx",
        additional_files=["onnx/model.onnx_data"],
    ),
]


class BuiltinSentenceEmbedding(OnnxTextEmbedding):
    """Builtin Sentence Embedding uses built-in pooling and normalization of underlying onnx models"""

    @classmethod
    def _get_worker_class(cls) -> Type[OnnxTextEmbeddingWorker]:
        return BuiltinSentenceEmbeddingWorker

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        """Lists the supported models.

        Returns:
            list[DenseModelDescription]: A list of DenseModelDescription objects containing the model information.
        """
        return supported_builtin_sentence_embedding_models

    def _load_tokenizer(self, model_dir: Path) -> None:
        self.tokenizer, self.special_token_to_id = load_tokenizer(
            model_dir=model_dir,
            default_max_length=DEFAULT_MAX_LENGTHS.get(self.model_name.lower()),
        )

    def _preprocess_onnx_input(
        self, onnx_input: dict[str, NumpyArray], **kwargs: Any
    ) -> dict[str, NumpyArray]:
        """Feed empty modality inputs to multimodal graphs used for text only.

        The embeddinggemma-2 export is a single graph that also declares `image_features`,
        `video_features` and `audio_features` inputs. For text, each is a zero-row tensor.
        """
        for node in self.model.get_inputs():  # type: ignore[union-attr]
            if node.name in onnx_input or not node.name.endswith("_features"):
                continue
            width = node.shape[-1] if isinstance(node.shape[-1], int) else 0
            onnx_input[node.name] = np.zeros((0, width), dtype=np.float32)
        return onnx_input

    def _post_process_onnx_output(
        self, output: OnnxOutputContext, **kwargs: Any
    ) -> Iterable[NumpyArray]:
        return output.model_output

    def _run_model(
        self, onnx_input: dict[str, Any], onnx_output_names: list[str] | None = None
    ) -> NumpyArray:
        return self.model.run(onnx_output_names, onnx_input)[1]  # type: ignore[union-attr]


class BuiltinSentenceEmbeddingWorker(OnnxTextEmbeddingWorker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs: Any,
    ) -> OnnxTextEmbedding:
        return BuiltinSentenceEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
