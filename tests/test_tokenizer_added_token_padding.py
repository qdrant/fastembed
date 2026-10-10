import json
from pathlib import Path
from typing import Any

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from fastembed.common.preprocessor_utils import load_tokenizer


def write_tokenizer(model_dir: Path, pad_token: str | dict[str, Any]) -> None:
    """Write a minimal tokenizer with padding defined only in its config."""
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2, "[PAD]": 3}))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(model_dir / "tokenizer.json"))
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 8, "pad_token": pad_token}), encoding="utf-8"
    )


@pytest.mark.parametrize(
    "pad_token",
    ["[PAD]", {"content": "[PAD]", "special": True, "normalized": False, "__type": "AddedToken"}],
)
def test_config_pad_token_accepts_string_and_added_token(tmp_path: Path, pad_token: Any) -> None:
    """Both supported token representations must pad the shorter sequence."""
    write_tokenizer(tmp_path, pad_token)

    tokenizer, _ = load_tokenizer(tmp_path)
    encoded = tokenizer.encode_batch(["hello world", "hello"])

    assert encoded[1].ids == [1, 3]
    assert encoded[1].attention_mask == [1, 0]
    assert tokenizer.padding is not None
    assert tokenizer.padding["pad_token"] == "[PAD]"
    assert tokenizer.padding["pad_id"] == 3


def test_serialized_padding_still_takes_precedence(tmp_path: Path) -> None:
    """Saved tokenizer padding must override conflicting config padding."""
    write_tokenizer(tmp_path, {"content": "[OTHER]", "special": True})
    tokenizer = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    tokenizer.enable_padding(pad_id=3, pad_token="[PAD]", direction="left")
    tokenizer.save(str(tmp_path / "tokenizer.json"))

    loaded, _ = load_tokenizer(tmp_path)
    encoded = loaded.encode_batch(["hello world", "hello"])

    assert encoded[1].ids == [3, 1]
    assert encoded[1].attention_mask == [0, 1]
