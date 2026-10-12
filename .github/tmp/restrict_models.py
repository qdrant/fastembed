"""pytest -p restrict_models: limits the tests to the int8 models of #803, temporary"""

from fastembed import ImageEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

KEEP = {
    "nomic-ai/nomic-embed-text-v1.5-Q",
    "nomic-ai/nomic-embed-vision-v1.5-Q",
    "BAAI/bge-reranker-v2-m3-int8",
}

for cls in (TextEmbedding, ImageEmbedding, TextCrossEncoder):
    original = cls._list_supported_models.__func__

    def restricted(c, original=original):
        return [m for m in original(c) if m.model in KEEP]

    cls._list_supported_models = classmethod(restricted)
