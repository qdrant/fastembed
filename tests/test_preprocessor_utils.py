import json
from pathlib import Path

import pytest

from fastembed.common.preprocessor_utils import load_tokenizer


def _write_tokenizer_dir(
    tmp_path: Path,
    *,
    model_max_length,
    max_length,
    include_model_max_length: bool = True,
    include_max_length: bool = True,
) -> Path:
    """
    Write the three files `load_tokenizer` requires, using a minimal but
    Tokenizers-loadable tokenizer.json.
    """
    (tmp_path / "config.json").write_text(json.dumps({"pad_token_id": 0}))

    tokenizer_config: dict = {"pad_token": "[PAD]"}
    if include_model_max_length:
        tokenizer_config["model_max_length"] = model_max_length
    if include_max_length:
        tokenizer_config["max_length"] = max_length
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))

    (tmp_path / "special_tokens_map.json").write_text(json.dumps({"pad_token": "[PAD]"}))

    # Minimal WordLevel tokenizer that the `tokenizers` crate accepts.
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": 0,
                "content": "[PAD]",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": {"[PAD]": 0, "hello": 1}, "unk_token": "[PAD]"},
    }
    (tmp_path / "tokenizer.json").write_text(json.dumps(tokenizer_json))

    return tmp_path


class TestLoadTokenizerMaxLength:
    """Regression tests for https://github.com/qdrant/fastembed/issues/685"""

    def test_zero_max_length_falls_back_to_model_max_length(self, tmp_path: Path) -> None:
        # `max_length=0` in a HuggingFace tokenizer config means "no cap" and
        # must not disable truncation. Prior behavior picked the 0 in the
        # `min()`, then crashed on any input longer than 0 tokens.
        _write_tokenizer_dir(
            tmp_path, model_max_length=512, max_length=0
        )
        tokenizer, _ = load_tokenizer(tmp_path)
        # `enable_truncation` should have been called with the non-zero cap.
        assert tokenizer.truncation is not None
        assert tokenizer.truncation["max_length"] == 512

    def test_zero_model_max_length_falls_back_to_max_length(self, tmp_path: Path) -> None:
        _write_tokenizer_dir(
            tmp_path, model_max_length=0, max_length=256
        )
        tokenizer, _ = load_tokenizer(tmp_path)
        assert tokenizer.truncation is not None
        assert tokenizer.truncation["max_length"] == 256

    def test_both_present_and_non_zero_takes_min(self, tmp_path: Path) -> None:
        _write_tokenizer_dir(
            tmp_path, model_max_length=512, max_length=256
        )
        tokenizer, _ = load_tokenizer(tmp_path)
        assert tokenizer.truncation is not None
        assert tokenizer.truncation["max_length"] == 256

    def test_only_max_length_present(self, tmp_path: Path) -> None:
        _write_tokenizer_dir(
            tmp_path,
            model_max_length=None,
            max_length=128,
            include_model_max_length=False,
        )
        tokenizer, _ = load_tokenizer(tmp_path)
        assert tokenizer.truncation is not None
        assert tokenizer.truncation["max_length"] == 128

    def test_only_model_max_length_present(self, tmp_path: Path) -> None:
        _write_tokenizer_dir(
            tmp_path,
            model_max_length=128,
            max_length=None,
            include_max_length=False,
        )
        tokenizer, _ = load_tokenizer(tmp_path)
        assert tokenizer.truncation is not None
        assert tokenizer.truncation["max_length"] == 128

    def test_both_zero_raises(self, tmp_path: Path) -> None:
        _write_tokenizer_dir(
            tmp_path, model_max_length=0, max_length=0
        )
        with pytest.raises(AssertionError):
            load_tokenizer(tmp_path)
