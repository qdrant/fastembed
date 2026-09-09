import json
from pathlib import Path
from typing import Any

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
    """
    Load and configure a tokenizer from a model directory.

    Configures truncation to the model context length, converts any fixed-length
    padding to dynamic batch padding (avoiding ragged batch failures), preserves
    serialized padding direction and token IDs, and optionally applies padding
    multiples (e.g. pad_to_multiple_of) when configured.

    Args:
        model_dir: Directory path containing tokenizer configuration files
            (config.json, tokenizer.json, tokenizer_config.json, special_tokens_map.json).

    Returns:
        A tuple of (configured Tokenizer instance, mapping of special token strings to token IDs).

    Raises:
        ValueError: If required configuration files are missing or if pad_to_multiple_of
            is not a positive integer.
    """
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
        assert "model_max_length" in tokenizer_config or "max_length" in tokenizer_config, (
            "Models without model_max_length or max_length are not supported."
        )
        if "model_max_length" not in tokenizer_config:
            max_context = tokenizer_config["max_length"]
        elif "max_length" not in tokenizer_config:
            max_context = tokenizer_config["model_max_length"]
        else:
            max_context = min(tokenizer_config["model_max_length"], tokenizer_config["max_length"])

    tokens_map = load_special_tokens(model_dir)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=max_context)

    pad_to_multiple_of = tokenizer_config.get("pad_to_multiple_of")
    if pad_to_multiple_of is None:
        pad_to_multiple_of = config.get("pad_to_multiple_of")

    if pad_to_multiple_of is not None and (
        not isinstance(pad_to_multiple_of, int)
        or isinstance(pad_to_multiple_of, bool)
        or pad_to_multiple_of <= 0
    ):
        raise ValueError("pad_to_multiple_of must be a positive integer")

    if not tokenizer.padding:
        tokenizer.enable_padding(
            pad_id=config.get("pad_token_id", 0),
            pad_token=tokenizer_config.get("pad_token", "[PAD]"),
            pad_to_multiple_of=pad_to_multiple_of,
        )
    else:
        padding_params = tokenizer.padding
        target_pad_to_multiple_of = (
            pad_to_multiple_of
            if pad_to_multiple_of is not None
            else padding_params.get("pad_to_multiple_of")
        )
        if target_pad_to_multiple_of is not None and (
            not isinstance(target_pad_to_multiple_of, int)
            or isinstance(target_pad_to_multiple_of, bool)
            or target_pad_to_multiple_of <= 0
        ):
            raise ValueError("pad_to_multiple_of must be a positive integer")

        if padding_params.get("length") is not None or (
            pad_to_multiple_of is not None
            and padding_params.get("pad_to_multiple_of") != target_pad_to_multiple_of
        ):
            tokenizer.enable_padding(
                direction=padding_params.get("direction", "right"),
                pad_id=padding_params.get("pad_id", config.get("pad_token_id", 0)),
                pad_type_id=padding_params.get("pad_type_id", 0),
                pad_token=padding_params.get(
                    "pad_token", tokenizer_config.get("pad_token", "[PAD]")
                ),
                pad_to_multiple_of=target_pad_to_multiple_of,
                length=None,
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
