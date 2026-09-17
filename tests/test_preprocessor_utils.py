import json
from pathlib import Path

from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordLevel

from fastembed.common.preprocessor_utils import load_tokenizer


def save_tokenizer(model_dir: Path, *, with_special_tokens: bool = False) -> None:
    tokenizer = Tokenizer(
        WordLevel(
            vocab={"[UNK]": 0, "[PAD]": 1, "[CLS]": 2, "hello": 3},
            unk_token="[UNK]",
        )
    )
    if with_special_tokens:
        tokenizer.add_special_tokens(
            [
                AddedToken("[PAD]", special=True),
                AddedToken("[CLS]", special=True),
            ]
        )
    tokenizer.save(str(model_dir / "tokenizer.json"))
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 16, "pad_token": "[PAD]"}),
        encoding="utf-8",
    )


def test_load_tokenizer_without_config(tmp_path: Path) -> None:
    save_tokenizer(tmp_path)
    (tmp_path / "special_tokens_map.json").write_text(
        json.dumps({"pad_token": "[PAD]", "cls_token": "[CLS]"}),
        encoding="utf-8",
    )

    tokenizer, special_token_to_id = load_tokenizer(tmp_path)

    assert tokenizer.padding is not None
    assert tokenizer.padding["pad_id"] == 1
    assert special_token_to_id == {"[PAD]": 1, "[CLS]": 2}


def test_load_tokenizer_without_special_tokens_map(tmp_path: Path) -> None:
    save_tokenizer(tmp_path, with_special_tokens=True)
    (tmp_path / "config.json").write_text(
        json.dumps({"pad_token_id": 1}),
        encoding="utf-8",
    )

    _, special_token_to_id = load_tokenizer(tmp_path)

    assert special_token_to_id == {"[PAD]": 1, "[CLS]": 2}
