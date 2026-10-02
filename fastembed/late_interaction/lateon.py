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
            "Text embeddings, Unimodal (text), English, 299 input tokens truncation, 2026 year"
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
    """LightOn's English ColBERT model, trained and exported with PyLate.

    It differs from colbert in two ways: a query is not expanded up to a fixed length with
    mask tokens, and the padding reuses the mask token id.

    TODO: PyLate writes `model_max_length` as its `document_length` minus the [D] marker, 299
    here, and colbert subtracts the marker a second time. Until the model repository reports
    the real limits, a document is cut at 299 tokens instead of 300, and a query is not cut at
    PyLate's `query_length` of 32 at all, which is how every other colbert model behaves here.
    """

    QUERY_MARKER_TOKEN_ID = 50368
    DOCUMENT_MARKER_TOKEN_ID = 50369
    MASK_TOKEN = "[MASK]"
    # exported with `do_query_expansion=false`, so a query is padded to the longest one in its
    # batch instead of to a fixed length, and that padding is dropped from the output
    MIN_QUERY_LENGTH = None

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

        # with `lazy_load` and `parallel`, inference runs in the workers and the parent
        # never calls `load_onnx_model`, so `skip_list` might not be set yet
        self._ensure_tokenizer()

        # padding reuses the mask token id, so it can only be told apart from a mask token in
        # the input by the attention mask, which is also what drops the padding from a query
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
    """The multilingual and code sibling of LateOn, based on mmBERT.

    Besides its own tokenizer and context length, it was trained with an empty skiplist, so
    punctuation is a part of a document rather than dropped from it.
    """

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
        self.skip_list = set()

        # TODO: drop once the model repository ships its own tokenizer metadata. It carries
        # LateOn's 299, and this model reads 8192 tokens, so without this every input is cut
        # at 299. The marker colbert inserts takes one of the 8192 positions.
        assert self.tokenizer is not None and self.query_tokenizer is not None
        self.tokenizer.enable_truncation(max_length=8192 - 1)
        self.query_tokenizer.enable_truncation(max_length=8192 - 1)


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
