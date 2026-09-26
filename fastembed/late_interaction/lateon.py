import string
from typing import Any, Iterable, Type

import numpy as np

from fastembed.common.model_description import DenseModelDescription, ModelSource
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.types import NumpyArray
from fastembed.common.preprocessor_utils import load_tokenizer
from fastembed.common.utils import iter_batch
from fastembed.late_interaction.colbert import Colbert, ColbertEmbeddingWorker


supported_lateon_models: list[DenseModelDescription] = [
    DenseModelDescription(
        model="lightonai/LateOn",
        dim=128,
        description=(
            "PyLate/ColBERT late-interaction English model based on ModernBERT, "
            "300 document tokens, 32 query tokens, 2025 year"
        ),
        license="apache-2.0",
        size_in_GB=0.616,
        sources=ModelSource(hf="lightonai/LateOn"),
        model_file="model.onnx",
        additional_files=["onnx_config.json"],
    ),
]


class LateOn(Colbert):
    QUERY_MARKER_TOKEN_ID = 50368
    DOCUMENT_MARKER_TOKEN_ID = 50369
    QUERY_LENGTH = 32
    DOCUMENT_LENGTH = 300
    MASK_TOKEN = "[MASK]"

    @classmethod
    def _get_worker_class(cls) -> Type[ColbertEmbeddingWorker]:
        return LateOnEmbeddingWorker

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        """Lists the supported LateOn models."""
        return supported_lateon_models

    def load_onnx_model(self) -> None:
        self._load_onnx_model(
            model_dir=self._model_dir,
            model_file=self.model_description.model_file,
            threads=self.threads,
            providers=self.providers,
            cuda=self.cuda,
            device_id=self.device_id,
            extra_session_options=self._extra_session_options,
        )
        self.query_tokenizer, _ = load_tokenizer(model_dir=self._model_dir)

        assert self.tokenizer is not None
        self.mask_token_id = self.special_token_to_id[self.MASK_TOKEN]
        self.pad_token_id = self.mask_token_id
        self.skip_list = {
            self.tokenizer.encode(symbol, add_special_tokens=False).ids[0]
            for symbol in string.punctuation
        }
        # LateOn's PyLate config uses document_length/query_length including the inserted
        # [D]/[Q] prefix token. Configure the tokenizer for the pre-prefix lengths.
        self.tokenizer.enable_truncation(max_length=self.DOCUMENT_LENGTH - 1)
        self.query_tokenizer.enable_truncation(max_length=self.QUERY_LENGTH - 1)

    def _post_process_onnx_output(
        self, output: OnnxOutputContext, is_doc: bool = True, **kwargs: Any
    ) -> Iterable[NumpyArray]:
        if is_doc:
            yield from super()._post_process_onnx_output(output, is_doc=is_doc, **kwargs)
            return

        if output.attention_mask is None:
            raise ValueError("attention_mask must be provided for query post-processing")

        for embedding, attention_mask in zip(output.model_output, output.attention_mask):
            # LateOn was exported with do_query_expansion=false, so query embeddings should
            # only include non-padding query tokens instead of ColBERT mask-token expansion.
            embedding = embedding[attention_mask == 1]
            norm = np.linalg.norm(embedding, ord=2, axis=1, keepdims=True)
            norm_clamped = np.maximum(norm, 1e-12)
            yield embedding / norm_clamped

    def token_count(
        self,
        texts: str | Iterable[str],
        batch_size: int = 1024,
        is_doc: bool = True,
        include_extension: bool = False,
        **kwargs: Any,
    ) -> int:
        if is_doc:
            return super().token_count(
                texts,
                batch_size=batch_size,
                is_doc=is_doc,
                include_extension=include_extension,
                **kwargs,
            )

        if not hasattr(self, "model") or self.model is None:
            self.load_onnx_model()

        token_num = 0
        texts = [texts] if isinstance(texts, str) else texts
        assert self.query_tokenizer is not None
        for batch in iter_batch(texts, batch_size):
            for tokens in self.query_tokenizer.encode_batch(batch):
                token_num += sum(tokens.attention_mask)
            if include_extension:
                token_num += len(batch)  # add one [Q] prefix token per query

        return token_num


class LateOnEmbeddingWorker(ColbertEmbeddingWorker):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs: Any) -> LateOn:
        return LateOn(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
