import json
from typing import Any
from pathlib import Path

from tokenizers import AddedToken, Tokenizer

from fastembed.image.transform.operators import Compose


def load_special_tokens(model_dir: Path) -> dict[str, Any]:
    tokens_map_path = model_dir / "special_tokens_map.json"
    if not tokens_map_path.exists():
        raise ValueError(f"Could not find special_tokens_map.json in {model_dir}")

    with open(str(tokens_map_path)) as tokens_map_file:
        tokens_map = json.load(tokens_map_file)

    return tokens_map


def load_tokenizer(model_dir: Path) -> tuple[Tokenizer, dict[str, int]]:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise ValueError(f"Could not find config.json in {model_dir}")

    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise ValueError(f"Could not find tokenizer.json in {model_dir}")

    tokenizer_config_path = model_dir / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        raise ValueError(f"Could not find tokenizer_config.json in {model_dir}")

    with open(str(config_path)) as config_file:
        config = json.load(config_file)

    with open(str(tokenizer_config_path)) as tokenizer_config_file:
        tokenizer_config = json.load(tokenizer_config_file)
        # HuggingFace tokenizer configs may set `model_max_length` or `max_length`
        # to 0 (or omit them entirely) — both mean "no cap". Truthiness handles
        # missing, None, and 0 uniformly so we don't disable truncation on
        # unusual-but-valid tokenizer configs.
        model_max_length = tokenizer_config.get("model_max_length")
        max_length = tokenizer_config.get("max_length")
        assert model_max_length or max_length, (
            "Models without a non-zero model_max_length or max_length are not supported."
        )
        if not model_max_length:
            max_context = max_length
        elif not max_length:
            max_context = model_max_length
        else:
            max_context = min(model_max_length, max_length)

    tokens_map = load_special_tokens(model_dir)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=max_context)
    if not tokenizer.padding:
        tokenizer.enable_padding(
            pad_id=config.get("pad_token_id", 0), pad_token=tokenizer_config["pad_token"]
        )

    for token in tokens_map.values():
        if isinstance(token, str):
            tokenizer.add_special_tokens([token])
        elif isinstance(token, dict):
            tokenizer.add_special_tokens([AddedToken(**token)])

    special_token_to_id: dict[str, int] = {}

    for token in tokens_map.values():
        if isinstance(token, str):
            special_token_to_id[token] = tokenizer.token_to_id(token)
        elif isinstance(token, dict):
            token_str = token.get("content", "")
            special_token_to_id[token_str] = tokenizer.token_to_id(token_str)

    return tokenizer, special_token_to_id


def load_preprocessor(model_dir: Path) -> Compose:
    preprocessor_config_path = model_dir / "preprocessor_config.json"
    if not preprocessor_config_path.exists():
        raise ValueError(f"Could not find preprocessor_config.json in {model_dir}")

    with open(str(preprocessor_config_path)) as preprocessor_config_file:
        preprocessor_config = json.load(preprocessor_config_file)
        transforms = Compose.from_config(preprocessor_config)
    return transforms
