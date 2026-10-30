"""Regression: load_static_tokenizer preserves the tokenizer's truncation direction.

The ``tokenizers`` library resets ``truncation.direction`` to ``"right"`` on every
``enable_truncation`` call.  :func:`load_static_tokenizer` must pass the configured
direction through instead of discarding it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from fastembed.text.static_embedding import MAX_LENGTH, load_static_tokenizer

N_TOKENS = 600
VOCAB: dict[str, int] = {"[UNK]": 0}
for i in range(1, N_TOKENS + 1):
    VOCAB[f"tok{i}"] = i

LONG_TEXT = " ".join(f"tok{i}" for i in range(1, N_TOKENS + 1))


def _write_tokenizer(model_dir: Path, *, direction: str | None) -> None:
    tok = Tokenizer(models.WordLevel(vocab=VOCAB, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    if direction is not None:
        tok.enable_truncation(max_length=MAX_LENGTH, direction=direction)
    tok.save(str(model_dir / "tokenizer.json"))


@pytest.mark.parametrize("direction", ["left", "right"])
def test_direction_dict_preserved(tmp_path: Path, direction: str) -> None:
    _write_tokenizer(tmp_path, direction=direction)
    tok, _ = load_static_tokenizer(tmp_path)
    assert tok.truncation is not None
    assert tok.truncation["direction"] == direction


def test_left_truncation_keeps_tail_tokens(tmp_path: Path) -> None:
    _write_tokenizer(tmp_path, direction="left")
    tok, _ = load_static_tokenizer(tmp_path)
    tokens = tok.encode(LONG_TEXT).tokens
    assert len(tokens) == MAX_LENGTH
    assert tokens[0] == "tok89"
    assert tokens[-1] == "tok600"


def test_right_truncation_keeps_head_tokens(tmp_path: Path) -> None:
    _write_tokenizer(tmp_path, direction="right")
    tok, _ = load_static_tokenizer(tmp_path)
    tokens = tok.encode(LONG_TEXT).tokens
    assert len(tokens) == MAX_LENGTH
    assert tokens[0] == "tok1"
    assert tokens[-1] == "tok512"


def test_no_truncation_defaults_to_right(tmp_path: Path) -> None:
    _write_tokenizer(tmp_path, direction=None)
    tok, _ = load_static_tokenizer(tmp_path)
    assert tok.truncation is not None
    assert tok.truncation["max_length"] == MAX_LENGTH
    assert tok.truncation["direction"] == "right"


def test_no_padding_is_applied(tmp_path: Path) -> None:
    _write_tokenizer(tmp_path, direction="left")
    tok, _ = load_static_tokenizer(tmp_path)
    assert tok.padding is None


def test_save_reload_roundtrip_preserves_left(tmp_path: Path) -> None:
    _write_tokenizer(tmp_path, direction="left")
    tok, _ = load_static_tokenizer(tmp_path)
    tok.save(str(tmp_path / "tokenizer.json"))
    tok2, _ = load_static_tokenizer(tmp_path)
    assert tok2.truncation["direction"] == "left"
