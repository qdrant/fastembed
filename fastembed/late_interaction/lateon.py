from pathlib import Path
from typing import Any, Iterable, Type

import numpy as np

from fastembed.common.model_description import DenseModelDescription, ModelSource
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.types import NumpyArray
from fastembed.late_interaction.colbert import Colbert, ColbertEmbeddingWorker


supported_lateon_models: list[DenseModelDescription] = [
    DenseModelDescription(
        model="lightonai/LateOn",
        dim=128,
        description=(
            "Text embeddings, Unimodal (text), English, 300 input tokens truncation, 2026 year"
        ),
        license="apache-2.0",
        size_in_GB=0.616,
        sources=ModelSource(hf="lightonai/LateOn"),
        model_file="model.onnx",
    ),
]

supported_mlateon_models: list[DenseModelDescription] = [
    DenseModelDescription(
        model="lightonai/mLateOn",
        dim=128,
        description=(
            "Text embeddings, Unimodal (text), Multilingual and code, "
            "8192 input tokens truncation, 2026 year"
        ),
        license="apache-2.0",
        size_in_GB=1.25,
        sources=ModelSource(hf="lightonai/mLateOn"),
        model_file="model.onnx",
    ),
]


class LateOn(Colbert):
    QUERY_MARKER_TOKEN_ID = 50368
    DOCUMENT_MARKER_TOKEN_ID = 50369
    MASK_TOKEN = "[MASK]"
    # exported with `do_query_expansion=false`
    MIN_QUERY_LENGTH = None
    # PyLate writes `model_max_length` with the [Q]/[D] marker already taken off
    RESERVED_MARKER_TOKENS = 0

    @classmethod
    def _get_worker_class(cls) -> Type[ColbertEmbeddingWorker]:
        return LateOnEmbeddingWorker

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        """Lists the supported LateOn models.

        Returns:
            list[DenseModelDescription]: A list of DenseModelDescription objects containing the model information.
        """
        return supported_lateon_models

    def _post_process_onnx_output(
        self, output: OnnxOutputContext, is_doc: bool = True, **kwargs: Any
    ) -> Iterable[NumpyArray]:
        if output.attention_mask is None:
            raise ValueError("attention_mask must be provided for post-processing")

        # inference can run in a worker, leaving `skip_list` unset in this process
        self._ensure_tokenizer()

        # padding reuses the mask token id, only the attention mask tells them apart
        keep = output.attention_mask == 1
        if is_doc and self.skip_list:
            if output.input_ids is None:
                raise ValueError("input_ids must be provided for document post-processing")
            keep &= ~np.isin(output.input_ids, list(self.skip_list))

        for embedding, keep_tokens in zip(output.model_output, keep):
            embedding = embedding[keep_tokens]
            norm = np.linalg.norm(embedding, ord=2, axis=1, keepdims=True)
            norm_clamped = np.maximum(norm, 1e-12)
            yield embedding / norm_clamped


class MLateOn(LateOn):
    QUERY_MARKER_TOKEN_ID = 256000
    DOCUMENT_MARKER_TOKEN_ID = 256001
    MASK_TOKEN = "<mask>"

    @classmethod
    def _get_worker_class(cls) -> Type[ColbertEmbeddingWorker]:
        return MLateOnEmbeddingWorker

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        """Lists the supported mLateOn models.

        Returns:
            list[DenseModelDescription]: A list of DenseModelDescription objects containing the model information.
        """
        return supported_mlateon_models

    def _load_tokenizer(self, model_dir: Path) -> None:
        super()._load_tokenizer(model_dir)
        # trained with an empty skiplist, punctuation is a part of a document
        self.skip_list = set()


class LateOnEmbeddingWorker(ColbertEmbeddingWorker):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs: Any) -> LateOn:
        return LateOn(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )


class MLateOnEmbeddingWorker(ColbertEmbeddingWorker):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs: Any) -> MLateOn:
        return MLateOn(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
