import itertools
import json
import os
import shutil

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tokenizers import Tokenizer

from fastembed.common.preprocessor_utils import load_tokenizer
from fastembed.text.text_embedding import TextEmbedding
from tests.utils import delete_model_cache

# transformers writes its VERY_LARGE_INTEGER in place of `model_max_length` when the real
# value is unknown, which is more than `enable_truncation` can accept
HF_SENTINEL = int(1e30)

# a lightweight model whose config files serve as a realistic starting point for the cases below
BASE_MODEL = "BAAI/bge-small-en-v1.5"
TOKENIZER_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def _patch_json(path: Path, overrides: dict[str, Any], drop: tuple[str, ...] = ()) -> None:
    with open(path) as source:
        content = json.load(source)

    content.update(overrides)
    for key in drop:
        content.pop(key, None)

    with open(path, "w") as target:
        json.dump(content, target)


def _set_serialized_padding(path: Path, padding: dict[str, Any] | None) -> None:
    """Rewrite tokenizer.json through the tokenizers library, so the format stays authoritative."""
    tokenizer = Tokenizer.from_file(str(path))
    if padding is None:
        tokenizer.no_padding()
    else:
        tokenizer.enable_padding(**padding)
    tokenizer.save(str(path))


@pytest.fixture(scope="module")
def make_model_dir(tmp_path_factory):
    """Build model directories from a real model's config files, with targeted overrides.

    `load_tokenizer` reads only the four files in `TOKENIZER_FILES`, so the onnx weights are
    never copied.
    """
    is_ci = os.getenv("CI")
    base_model = TextEmbedding(BASE_MODEL)
    source_dir = Path(base_model.model._model_dir)
    counter = itertools.count()

    def factory(
        tokenizer_config: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        padding: dict[str, Any] | None = None,
        drop_from_tokenizer_config: tuple[str, ...] = (),
        drop_from_config: tuple[str, ...] = (),
        drop_files: tuple[str, ...] = (),
    ) -> Path:
        model_dir = tmp_path_factory.mktemp(f"model_dir_{next(counter)}")
        for file_name in TOKENIZER_FILES:
            if file_name in drop_files:
                continue
            shutil.copy(source_dir / file_name, model_dir / file_name)

        if "tokenizer_config.json" not in drop_files:
            _patch_json(
                model_dir / "tokenizer_config.json",
                tokenizer_config or {},
                drop_from_tokenizer_config,
            )
        if "config.json" not in drop_files:
            _patch_json(model_dir / "config.json", config or {}, drop_from_config)
        if padding is not None:
            _set_serialized_padding(model_dir / "tokenizer.json", padding)

        return model_dir

    yield factory

    if is_ci:
        delete_model_cache(base_model.model._model_dir)


def test_fixed_padding_is_relaxed_to_batch_longest(make_model_dir) -> None:
    """Fixed padding shorter than the truncation limit leaves longer encodings ragged."""
    model_dir = make_model_dir(
        tokenizer_config={"model_max_length": 512, "max_length": None},
        padding={"length": 128, "pad_id": 0, "pad_token": "[PAD]", "direction": "right"},
    )

    tokenizer, _ = load_tokenizer(model_dir)

    assert tokenizer.padding["length"] is None
    assert tokenizer.truncation["max_length"] == 512

    encoded = tokenizer.encode_batch(["hello world", "retrieval " * 200])
    # ragged encodings make this raise, the same way onnx_embed does
    input_ids = np.array([encoding.ids for encoding in encoded])

    assert input_ids.shape == (2, len(encoded[0].ids))
    assert input_ids.shape[1] > 128


def test_batch_longest_padding_does_not_pad_to_the_truncation_limit(make_model_dir) -> None:
    model_dir = make_model_dir(
        tokenizer_config={"model_max_length": 512, "max_length": None},
        padding={"length": 128, "pad_id": 0, "pad_token": "[PAD]", "direction": "right"},
    )

    tokenizer, _ = load_tokenizer(model_dir)
    encoded = tokenizer.encode_batch(["hello world", "hello"])

    assert len(encoded[0].ids) == len(encoded[1].ids) < 128


def test_serialized_left_padding_is_preserved(make_model_dir) -> None:
    """ColModernVBERT pads on the left, normalizing the length must not reset the direction."""
    model_dir = make_model_dir(
        padding={"length": None, "pad_id": 0, "pad_token": "[PAD]", "direction": "left"},
    )

    tokenizer, _ = load_tokenizer(model_dir)

    assert tokenizer.padding["direction"] == "left"
    assert tokenizer.padding["length"] is None

    encoded = tokenizer.encode_batch(["hello world and then some", "hello"])
    assert encoded[1].ids[0] == 0
    assert encoded[1].attention_mask[0] == 0


def test_serialized_pad_to_multiple_of_is_preserved(make_model_dir) -> None:
    """Everything the tokenizer declared is kept; only the fixed length is overridden."""
    model_dir = make_model_dir(
        padding={"length": 128, "pad_id": 0, "pad_token": "[PAD]", "pad_to_multiple_of": 8},
    )

    tokenizer, _ = load_tokenizer(model_dir)

    assert tokenizer.padding["length"] is None
    assert tokenizer.padding["pad_to_multiple_of"] == 8

    encoded = tokenizer.encode_batch(["hello world", "hello"])
    assert len(encoded[0].ids) % 8 == 0


def test_serialized_pad_id_takes_precedence_over_config(make_model_dir) -> None:
    mask_pad_id = 103  # [MASK] in the bert-base vocab, any id other than the config's works
    model_dir = make_model_dir(
        config={"pad_token_id": 7},
        padding={"length": None, "pad_id": mask_pad_id, "pad_token": "[MASK]"},
    )

    tokenizer, _ = load_tokenizer(model_dir)

    assert tokenizer.padding["pad_id"] == mask_pad_id
    assert tokenizer.padding["pad_token"] == "[MASK]"


def test_pad_token_falls_back_to_tokenizer_config(make_model_dir) -> None:
    model_dir = make_model_dir(config={"pad_token_id": 3}, padding=None)

    tokenizer, _ = load_tokenizer(model_dir)

    assert tokenizer.padding["pad_token"] == "[PAD]"
    assert tokenizer.padding["pad_id"] == 3


def test_missing_pad_token_raises(make_model_dir) -> None:
    model_dir = make_model_dir(drop_from_tokenizer_config=("pad_token",))

    with pytest.raises(ValueError, match="Could not find a pad token"):
        load_tokenizer(model_dir)


@pytest.mark.parametrize(
    "model_max_length,max_length,expected",
    [
        (512, 128, 128),  # both usable, the stricter one wins
        (128, 512, 128),
        (512, None, 512),
        (None, 256, 256),
        (HF_SENTINEL, 128, 128),  # qdrant/gte-large-onnx
        (0, 256, 256),
        (512, 0, 512),
    ],
)
def test_max_context_resolution(make_model_dir, model_max_length, max_length, expected) -> None:
    model_dir = make_model_dir(
        tokenizer_config={"model_max_length": model_max_length, "max_length": max_length},
    )

    tokenizer, _ = load_tokenizer(model_dir)

    assert tokenizer.truncation["max_length"] == expected


@pytest.mark.parametrize(
    "model_max_length,max_length",
    [
        (HF_SENTINEL, None),  # transformers' placeholder is not a limit
        (0, None),  # a zero would truncate everything away
        (None, 0),
        (None, None),
        ("512", None),  # not an integer
    ],
)
def test_unusable_max_context_raises(make_model_dir, model_max_length, max_length) -> None:
    model_dir = make_model_dir(
        tokenizer_config={"model_max_length": model_max_length, "max_length": max_length},
    )

    with pytest.raises(ValueError, match="Could not determine the maximum context length"):
        load_tokenizer(model_dir)


def test_absent_max_context_keys_raise(make_model_dir) -> None:
    model_dir = make_model_dir(
        drop_from_tokenizer_config=("model_max_length", "max_length"),
    )

    with pytest.raises(ValueError, match="Could not determine the maximum context length"):
        load_tokenizer(model_dir)


@pytest.fixture(scope="module")
def token_id(make_model_dir):
    """Resolve vocabulary ids by name, so the cases below carry no magic numbers."""
    return Tokenizer.from_file(str(make_model_dir() / "tokenizer.json")).token_to_id


@pytest.mark.parametrize(
    "dropped",
    [("config.json",), ("special_tokens_map.json",), ("config.json", "special_tokens_map.json")],
    ids=["no-config", "no-special-tokens-map", "neither"],
)
def test_optional_files_do_not_change_what_is_loaded(make_model_dir, dropped) -> None:
    """Both files are redundant: everything they carry is already in the tokenizer."""
    baseline, baseline_specials = load_tokenizer(make_model_dir())

    tokenizer, specials = load_tokenizer(make_model_dir(drop_files=dropped))

    assert specials == baseline_specials
    assert tokenizer.padding == baseline.padding
    assert tokenizer.encode("hello world").ids == baseline.encode("hello world").ids


@pytest.mark.parametrize("missing", ("tokenizer.json", "tokenizer_config.json"))
def test_the_remaining_files_are_still_required(make_model_dir, missing) -> None:
    """Relaxing the optional two must not relax the two that carry irreplaceable data."""
    model_dir = make_model_dir(drop_files=(missing,))

    with pytest.raises(ValueError, match=f"Could not find {missing}"):
        load_tokenizer(model_dir)


@pytest.mark.parametrize(
    "model_files",
    [
        pytest.param({"drop_from_config": ("pad_token_id",)}, id="config-omits-pad-token-id"),
        pytest.param({"drop_files": ("config.json",)}, id="config-is-absent"),
    ],
)
def test_pad_id_falls_back_to_the_vocabulary(make_model_dir, token_id, model_files) -> None:
    """Last link of the chain; a hardcoded 0 would silently disagree with `pad_token`."""
    expected = token_id("[SEP]")
    assert expected != 0, "a pad token whose id is 0 would pass even without a lookup"
    model_dir = make_model_dir(tokenizer_config={"pad_token": "[SEP]"}, **model_files)

    tokenizer, _ = load_tokenizer(model_dir)

    assert tokenizer.padding["pad_id"] == expected


def test_pad_token_that_resolves_nowhere_raises(make_model_dir) -> None:
    """Without config.json a pad token outside the vocabulary has no id left to fall back on."""
    model_dir = make_model_dir(
        tokenizer_config={"pad_token": "[NOT_IN_VOCAB]"},
        drop_files=("config.json",),
    )

    with pytest.raises(ValueError, match="Could not resolve an id for the pad token"):
        load_tokenizer(model_dir)


def test_pad_token_named_only_in_the_map_resolves(make_model_dir) -> None:
    """The map is read first, so it can name a pad token tokenizer.json does not carry."""
    model_dir = make_model_dir(
        tokenizer_config={"pad_token": "<|mypad|>"},
        drop_files=("config.json",),
    )
    _patch_json(model_dir / "special_tokens_map.json", {"pad_token": "<|mypad|>"})

    tokenizer, specials = load_tokenizer(model_dir)

    assert tokenizer.padding["pad_token"] == "<|mypad|>"
    assert tokenizer.padding["pad_id"] == specials["<|mypad|>"]


@pytest.mark.parametrize(
    "additional",
    [
        pytest.param(["<|list_str|>"], id="list-of-strings"),
        pytest.param([{"content": "<|list_str|>"}], id="list-of-added-token-dicts"),
    ],
)
def test_list_valued_map_entries_are_registered(make_model_dir, additional) -> None:
    """`additional_special_tokens` holds a list, which the str/dict dispatch alone drops.

    Real repos ship both spellings, and their tokens are in tokenizer.json already, so
    only a token living nowhere else shows the drop.
    """
    model_dir = make_model_dir()
    _patch_json(model_dir / "special_tokens_map.json", {"additional_special_tokens": additional})

    _, specials = load_tokenizer(model_dir)

    assert "<|list_str|>" in specials
