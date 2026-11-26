import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models

from fastembed import (
    ImageEmbedding,
    LateInteractionMultimodalEmbedding,
    LateInteractionTextEmbedding,
    SparseTextEmbedding,
    TextEmbedding,
)
from fastembed.rerank.cross_encoder import TextCrossEncoder


@pytest.mark.filterwarnings(
    "ignore:The model 'Qdrant/bm42-all-minilm-l6-v2-attentions':UserWarning"
)
@pytest.mark.parametrize(
    "model_class,model_name",
    [
        (TextEmbedding, "BAAI/bge-small-en-v1.5"),
        (ImageEmbedding, "Qdrant/clip-ViT-B-32-vision"),
        (LateInteractionTextEmbedding, "colbert-ir/colbertv2.0"),
        (LateInteractionMultimodalEmbedding, "Qdrant/colmodernvbert"),
        (LateInteractionMultimodalEmbedding, "Qdrant/colpali-v1.3-fp16"),
        (TextCrossEncoder, "Xenova/ms-marco-MiniLM-L-6-v2"),
        (SparseTextEmbedding, "Qdrant/bm42-all-minilm-l6-v2-attentions"),
        (SparseTextEmbedding, "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte"),
        (SparseTextEmbedding, "prithivida/Splade_PP_en_v1"),
    ],
)
@pytest.mark.parametrize(
    "device_ids,device_id,expected",
    [
        ([], None, None),
        ([], 0, 0),
        (None, None, None),
        ([2, 1], None, 2),
        ([2, 1], 1, 1),
    ],
    ids=["empty", "empty-with-explicit-zero", "omitted", "first", "explicit"],
)
def test_onnx_device_selection(
    model_class,
    model_name: str,
    device_ids: list[int] | None,
    device_id: int | None,
    expected: int | None,
    tmp_path: Path,
) -> None:
    """Select a device during lazy construction without an ONNX model or GPU."""
    if model_name == "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte":
        # IF-SPLADE loads its query tokenizer and IDF table even with lazy_load=True.
        vocab = {"[PAD]": 0, "[UNK]": 1}
        tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
        tokenizer.save(str(tmp_path / "tokenizer.json"))
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"model_max_length": 8, "pad_token": "[PAD]"}), encoding="utf-8"
        )
        (tmp_path / "idf.json").write_text("{}", encoding="utf-8")

    embedding = model_class(
        model_name,
        cache_dir=str(tmp_path),
        specific_model_path=str(tmp_path),
        local_files_only=True,
        lazy_load=True,
        cuda=False,
        device_ids=device_ids,
        device_id=device_id,
    )

    assert embedding.model.device_id == expected
