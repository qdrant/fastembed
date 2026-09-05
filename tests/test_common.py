import numpy as np

from fastembed import (
    ImageEmbedding,
    LateInteractionMultimodalEmbedding,
    LateInteractionTextEmbedding,
    SparseTextEmbedding,
    TextEmbedding,
)
from fastembed.common.utils import last_token_pooling


def test_text_list_supported_models():
    for model_type in [
        TextEmbedding,
        SparseTextEmbedding,
        ImageEmbedding,
        LateInteractionMultimodalEmbedding,
        LateInteractionTextEmbedding,
    ]:
        supported_models = model_type.list_supported_models()
        assert isinstance(supported_models, list)
        description = supported_models[0]
        assert isinstance(description, dict)

        assert description.get("model")
        if model_type != SparseTextEmbedding:
            assert description.get("dim")
        assert description.get("license")
        assert description.get("size_in_GB")
        assert description.get("model_file")
        assert description.get("sources")
        assert "hf" in description["sources"] or "url" in description["sources"]


def test_last_token_pooling():
    token_embeddings = np.array(
        [
            [[1.0, 1.0], [2.0, 2.0], [9.0, 9.0], [9.0, 9.0]],  # 2 real tokens, then padding
            [[3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],  # no padding
        ]
    )
    attention_mask = np.array([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=np.int64)

    pooled = last_token_pooling(token_embeddings, attention_mask)

    assert np.allclose(pooled, [[2.0, 2.0], [6.0, 6.0]])


def test_last_token_pooling_with_left_padding():
    token_embeddings = np.array(
        [
            [[9.0, 9.0], [9.0, 9.0], [1.0, 1.0], [2.0, 2.0]],  # padding, then 2 real tokens
            [[3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],  # no padding
        ]
    )
    attention_mask = np.array([[0, 0, 1, 1], [1, 1, 1, 1]], dtype=np.int64)

    pooled = last_token_pooling(token_embeddings, attention_mask)

    assert np.allclose(pooled, [[2.0, 2.0], [6.0, 6.0]])


def test_load_tokenizer_fixed_length_padding_converted_to_dynamic(tmp_path):
    import json

    from tokenizers import Tokenizer, models

    from fastembed.common.preprocessor_utils import load_tokenizer

    config = {"pad_token_id": 0}
    with open(tmp_path / "config.json", "w") as f:
        json.dump(config, f)

    tokenizer_config = {
        "model_max_length": 512,
        "pad_token": "[PAD]",
    }
    with open(tmp_path / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f)

    with open(tmp_path / "special_tokens_map.json", "w") as f:
        json.dump({"pad_token": "[PAD]"}, f)

    # Tokenizer initialized with fixed-length padding (e.g. gte-base with length=128)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.add_special_tokens(["[PAD]"])
    tokenizer.enable_padding(length=128, pad_id=0, pad_token="[PAD]", direction="right")
    tokenizer.save(str(tmp_path / "tokenizer.json"))

    loaded_tokenizer, _ = load_tokenizer(tmp_path)

    # Fixed length must be relaxed to None (dynamic batch padding) to prevent ragged arrays
    assert loaded_tokenizer.padding is not None
    assert loaded_tokenizer.padding["length"] is None
    assert loaded_tokenizer.padding["direction"] == "right"
    assert loaded_tokenizer.padding["pad_id"] == 0
    assert loaded_tokenizer.padding["pad_token"] == "[PAD]"
    assert loaded_tokenizer.truncation["max_length"] == 512


def test_load_tokenizer_preserves_left_padding(tmp_path):
    import json

    from tokenizers import Tokenizer, models

    from fastembed.common.preprocessor_utils import load_tokenizer

    config = {"pad_token_id": 50283}
    with open(tmp_path / "config.json", "w") as f:
        json.dump(config, f)

    tokenizer_config = {
        "model_max_length": 8192,
        "pad_token": "[PAD]",
    }
    with open(tmp_path / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f)

    with open(tmp_path / "special_tokens_map.json", "w") as f:
        json.dump({"pad_token": "[PAD]"}, f)

    # Tokenizer with dynamic left padding (e.g. ColModernVBERT)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.add_special_tokens(["[PAD]"])
    tokenizer.enable_padding(length=None, pad_id=50283, pad_token="[PAD]", direction="left")
    tokenizer.save(str(tmp_path / "tokenizer.json"))

    loaded_tokenizer, _ = load_tokenizer(tmp_path)

    # Preserves left padding direction and pad_id
    assert loaded_tokenizer.padding is not None
    assert loaded_tokenizer.padding["length"] is None
    assert loaded_tokenizer.padding["direction"] == "left"
    assert loaded_tokenizer.padding["pad_id"] == 50283
