import hashlib

from fastembed import (
    TextEmbedding,
    LateInteractionTextEmbedding,
    SparseTextEmbedding,
    ImageEmbedding,
    LateInteractionMultimodalEmbedding,
)
from fastembed.rerank.cross_encoder import TextCrossEncoder


models = [
    *TextEmbedding.list_supported_models(),
    *ImageEmbedding.list_supported_models(),
    *LateInteractionTextEmbedding.list_supported_models(),
    *LateInteractionMultimodalEmbedding.list_supported_models(),
    *TextCrossEncoder.list_supported_models(),
    *SparseTextEmbedding.list_supported_models(),
]

model_names = sorted(set([model["model"] for model in models]))

hash_value = hashlib.sha256("".join(model_names).encode()).hexdigest()

print(hash_value)
