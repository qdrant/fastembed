import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from fastembed.common.preprocessor_utils import load_tokenizer


@pytest.mark.parametrize(
    ("strategy", "expected_lengths"),
    [("only_first", (2, 4)), ("only_second", (4, 2)), ("longest_first", (3, 3)), (None, (3, 3))],
)
def test_load_tokenizer_preserves_truncation_strategy(
    tmp_path: Path, strategy: str | None, expected_lengths: tuple[int, int]
) -> None:
    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "q": 2, "d": 3}))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    if strategy is not None:
        tokenizer.enable_truncation(max_length=7, strategy=strategy)
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 6}))

    loaded, _ = load_tokenizer(tmp_path)
    encoding = loaded.encode("q q q q", "d d d d")

    assert loaded.truncation["strategy"] == (strategy or "longest_first")
    assert loaded.truncation["max_length"] == 6
    assert (encoding.sequence_ids.count(0), encoding.sequence_ids.count(1)) == expected_lengths
