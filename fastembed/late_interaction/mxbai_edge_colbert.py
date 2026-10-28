from typing import Any, Iterable, Type

from tokenizers import Encoding

from fastembed.common.model_description import DenseModelDescription, ModelSource
from fastembed.late_interaction.colbert import ColbertEmbeddingWorker
from fastembed.late_interaction.lateon import LateOn


supported_mxbai_edge_colbert_models: list[DenseModelDescription] = [
    DenseModelDescription(
        model="mixedbread-ai/mxbai-edge-colbert-v0-17m",
        dim=48,
        description=(
            "Text embeddings, Unimodal (text), English, 8000 input tokens truncation, 2025 year"
        ),
        license="apache-2.0",
        size_in_GB=0.068,
        sources=ModelSource(hf="mixedbread-ai/mxbai-edge-colbert-v0-17m"),
        model_file="model.onnx",
    ),
]


class MxbaiEdgeColbert(LateOn):
    """Shares LateOn's tokenizer, markers and the lack of query expansion, but lowercases text"""

    @classmethod
    def _get_worker_class(cls) -> Type[ColbertEmbeddingWorker]:
        return MxbaiEdgeColbertEmbeddingWorker

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        """Lists the supported mxbai-edge-colbert models.

        Returns:
            list[DenseModelDescription]: A list of DenseModelDescription objects containing the model information.
        """
        return supported_mxbai_edge_colbert_models

    # sentence-transformers lowercases the raw text (`do_lower_case` in sentence_bert_config.json).
    # A tokenizer normalizer would differ: it runs after special tokens like [MASK] are matched,
    # and lowercases char by char, so a word-final Σ never becomes ς
    def tokenize(self, documents: list[str], is_doc: bool = True, **kwargs: Any) -> list[Encoding]:
        return super().tokenize([document.lower() for document in documents], is_doc, **kwargs)

    def token_count(
        self,
        texts: str | Iterable[str],
        batch_size: int = 1024,
        is_doc: bool = True,
        include_extension: bool = False,
        **kwargs: Any,
    ) -> int:
        texts = [texts] if isinstance(texts, str) else texts
        return super().token_count(
            (text.lower() for text in texts),
            batch_size=batch_size,
            is_doc=is_doc,
            include_extension=include_extension,
            **kwargs,
        )


class MxbaiEdgeColbertEmbeddingWorker(ColbertEmbeddingWorker):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs: Any) -> MxbaiEdgeColbert:
        return MxbaiEdgeColbert(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
