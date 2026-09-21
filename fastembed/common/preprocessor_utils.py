import json
import sys
from typing import Any, Iterator
from pathlib import Path

from tokenizers import AddedToken, Tokenizer

from fastembed.image.transform.operators import Compose


def load_special_tokens(model_dir: Path) -> dict[str, Any]:
    """Read special_tokens_map.json, treating an absent file as an empty map.

    Newer transformers releases stop writing the file, and everything it holds is also
    recorded in tokenizer.json, so its absence is not an error.
    """
    tokens_map_path = model_dir / "special_tokens_map.json"
    if not tokens_map_path.exists():
        return {}

    with open(str(tokens_map_path)) as tokens_map_file:
        tokens_map = json.load(tokens_map_file)

    return tokens_map


def iter_special_tokens(tokens_map: dict[str, Any]) -> Iterator[str | dict[str, Any]]:
    """Yield the individual tokens declared in a special tokens map.

    Most keys hold a single token, either a bare string or an `AddedToken` dict, but
    `additional_special_tokens` holds a list of them, which has to be flattened before
    the tokens can be dispatched on their type.
    """
    for value in tokens_map.values():
        if isinstance(value, list):
            yield from value
        else:
            yield value


def _valid_context(value: Any) -> int | None:
    """Return `value` if it can be used as a truncation limit, `None` otherwise.

    Config files do not always carry a real limit: transformers writes `model_max_length` as
    1e30 when the value is unknown, and some repos ship a 0 or a null. `enable_truncation`
    raises an `OverflowError` on the former and silently produces empty encodings on the
    latter, so both are rejected here rather than passed through.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if not 0 < value <= sys.maxsize:
        return None
    return value


def _resolve_max_context(tokenizer_config: dict[str, Any], model_dir: Path) -> int:
    """Pick the truncation limit, preferring the stricter of the two tokenizer config keys.

    `config.json:max_position_embeddings` deliberately is not used as a fallback: it is the size
    of the position table, not the usable context, and the two differ per architecture, e.g.
    roberta reports 514 for a usable 512.
    """
    candidates = [
        context
        for context in (
            _valid_context(tokenizer_config.get("model_max_length")),
            _valid_context(tokenizer_config.get("max_length")),
        )
        if context is not None
    ]
    if not candidates:
        raise ValueError(
            f"Could not determine the maximum context length for {model_dir}. Set a positive "
            "`model_max_length` or `max_length` in tokenizer_config.json."
        )

    return min(candidates)


def load_tokenizer(model_dir: Path) -> tuple[Tokenizer, dict[str, int]]:
    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise ValueError(f"Could not find tokenizer.json in {model_dir}")

    tokenizer_config_path = model_dir / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        raise ValueError(f"Could not find tokenizer_config.json in {model_dir}")

    # config.json is optional: it only ever contributes pad_token_id, and newer transformers
    # releases no longer write it for every model.
    config_path = model_dir / "config.json"
    config: dict[str, Any] = {}
    if config_path.exists():
        with open(str(config_path)) as config_file:
            config = json.load(config_file)

    with open(str(tokenizer_config_path)) as tokenizer_config_file:
        tokenizer_config = json.load(tokenizer_config_file)

    max_context = _resolve_max_context(tokenizer_config, model_dir)

    tokens_map = load_special_tokens(model_dir)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=max_context)

    # Special tokens are registered before the padding is resolved: the map may name a pad
    # token that tokenizer.json does not carry, and it only gets an id once it is added.
    for token in iter_special_tokens(tokens_map):
        if isinstance(token, str):
            tokenizer.add_special_tokens([token])
        elif isinstance(token, dict):
            tokenizer.add_special_tokens([AddedToken(**token)])

    # Padding is always normalized to batch-longest. A serialized fixed length shorter than the
    # truncation limit leaves longer encodings untouched, which produces ragged batches, and a
    # fixed length equal to it pads every batch to the maximum. Direction and pad token metadata
    # are taken from the serialized settings, since some models pad on the left.
    padding = tokenizer.padding or {}
    pad_token = padding.get("pad_token") or tokenizer_config.get("pad_token")
    if pad_token is None:
        raise ValueError(f"Could not find a pad token for {model_dir}")

    # `config.json` is optional, and even when it is present it does not always carry a
    # `pad_token_id`, so the vocabulary is the last resort. A hardcoded 0 is not: it silently
    # disagrees with `pad_token` for every model whose pad token is not the first entry.
    pad_id = padding.get("pad_id", config.get("pad_token_id"))
    if pad_id is None:
        pad_id = tokenizer.token_to_id(pad_token)
    if pad_id is None:
        raise ValueError(f"Could not resolve an id for the pad token {pad_token!r} in {model_dir}")

    tokenizer.enable_padding(
        direction=padding.get("direction", "right"),
        pad_id=pad_id,
        pad_type_id=padding.get("pad_type_id", 0),
        pad_token=pad_token,
        pad_to_multiple_of=padding.get("pad_to_multiple_of"),
        length=None,
    )

    special_token_to_id = {
        token.content: token_id
        for token_id, token in tokenizer.get_added_tokens_decoder().items()
        if token.special
    }

    return tokenizer, special_token_to_id


def load_preprocessor(model_dir: Path) -> Compose:
    preprocessor_config_path = model_dir / "preprocessor_config.json"
    if not preprocessor_config_path.exists():
        raise ValueError(f"Could not find preprocessor_config.json in {model_dir}")

    with open(str(preprocessor_config_path)) as preprocessor_config_file:
        preprocessor_config = json.load(preprocessor_config_file)
        transforms = Compose.from_config(preprocessor_config)
    return transforms
