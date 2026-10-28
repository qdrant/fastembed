from pathlib import Path
from typing import Any, Type

from tokenizers import normalizers

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

    def _load_tokenizer(self, model_dir: Path) -> None:
        super()._load_tokenizer(model_dir)
        # sentence-transformers lowercases the text (`do_lower_case` in sentence_bert_config.json),
        # tokenizer.json itself only applies NFC
        lowercase = normalizers.Lowercase()
        for tokenizer in (self.tokenizer, self.query_tokenizer):
            assert tokenizer is not None
            normalizer = tokenizer.normalizer
            tokenizer.normalizer = (
                normalizers.Sequence([normalizer, lowercase]) if normalizer else lowercase
            )


class MxbaiEdgeColbertEmbeddingWorker(ColbertEmbeddingWorker):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs: Any) -> MxbaiEdgeColbert:
        return MxbaiEdgeColbert(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
