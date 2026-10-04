"""Offline regression coverage for serialized tokenizer truncation direction."""

import json
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from fastembed.common.preprocessor_utils import load_tokenizer
from fastembed.text.onnx_text_model import OnnxTextModel


def make_tokenizer_dir(
    path: Path,
    direction: str | None,
    context: int,
    serialized_context: int = 5,
    max_length: int | None = None,
    stride: int = 0,
    padding_direction: str = "right",
) -> Path:
    """Save a tiny real tokenizer without downloading a model or starting ONNX."""
    tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "one", "two", "three", "four", "five"]
    tokenizer = Tokenizer(models.WordLevel(dict(zip(tokens, range(len(tokens)))), "[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    if direction is not None:
        tokenizer.enable_truncation(
            max_length=serialized_context, direction=direction, stride=stride
        )
    tokenizer.enable_padding(pad_id=1, pad_token="[PAD]", direction=padding_direction)
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": context, "max_length": max_length, "pad_token": "[PAD]"})
    )
    return path


@pytest.mark.parametrize("direction", ["left", "right", None])
@pytest.mark.parametrize("context", [4, 5, 6])
def test_load_tokenizer_preserves_truncation_direction(tmp_path, direction, context):
    """Changing the context limit must not change which end of the input survives."""
    model_dir = make_tokenizer_dir(tmp_path, direction, context)
    tokenizer, _ = load_tokenizer(model_dir)
    words = ["one", "two", "three", "four", "five"]
    kept_words = words[-(context - 2) :] if direction == "left" else words[: context - 2]

    assert tokenizer.truncation["max_length"] == context
    assert tokenizer.truncation["direction"] == (direction or "right")
    assert tokenizer.encode(" ".join(words)).tokens == ["[CLS]", *kept_words, "[SEP]"]


@pytest.mark.parametrize("direction", ["left", "right"])
def test_text_model_uses_preserved_direction_in_mixed_length_batch(tmp_path, direction):
    """The real text-model tokenizer retains the declared text and still pads batches."""
    model_dir = make_tokenizer_dir(tmp_path, direction, context=5)
    model = OnnxTextModel()
    model._load_tokenizer(model_dir)

    encoded = model.tokenize(["one two three four five", "one"])
    expected_long = ["three", "four", "five"] if direction == "left" else ["one", "two", "three"]
    assert encoded[0].tokens == ["[CLS]", *expected_long, "[SEP]"]
    assert encoded[1].tokens == ["[CLS]", "one", "[SEP]", "[PAD]", "[PAD]"]
    assert encoded[1].attention_mask == [1, 1, 1, 0, 0]
    assert np.array([encoding.ids for encoding in encoded]).shape == (2, 5)


@pytest.mark.parametrize("context,max_length,expected_length", [(8, 5, 5), (4, 6, 4)])
def test_saved_direction_keeps_the_stricter_config_limit(
    tmp_path, context, max_length, expected_length
):
    """Direction preservation does not replace the loader's context-limit resolution."""
    model_dir = make_tokenizer_dir(
        tmp_path, "left", context, serialized_context=8, max_length=max_length
    )
    tokenizer, _ = load_tokenizer(model_dir)
    words = ["one", "two", "three", "four", "five"]

    assert tokenizer.truncation["max_length"] == expected_length
    assert tokenizer.encode(" ".join(words)).tokens == [
        "[CLS]",
        *words[-(expected_length - 2) :],
        "[SEP]",
    ]


def test_saved_direction_does_not_restore_stride(tmp_path):
    """The loader still disables overflow stride when applying a smaller context."""
    model_dir = make_tokenizer_dir(tmp_path, "left", context=4, serialized_context=8, stride=3)

    tokenizer, _ = load_tokenizer(model_dir)

    assert tokenizer.truncation["direction"] == "left"
    assert tokenizer.truncation["stride"] == 0
    assert tokenizer.truncation["strategy"] == "longest_first"
    assert tokenizer.encode("one two three four five").tokens == ["[CLS]", "four", "five", "[SEP]"]


def test_left_truncation_and_left_padding_remain_independent(tmp_path):
    """Padding keeps short items aligned without changing the long item's retained end."""
    model_dir = make_tokenizer_dir(tmp_path, "left", context=5, padding_direction="left")
    tokenizer, _ = load_tokenizer(model_dir)

    encoded = tokenizer.encode_batch(["one two three four five", "one"])

    assert encoded[0].tokens == ["[CLS]", "three", "four", "five", "[SEP]"]
    assert encoded[1].tokens == ["[PAD]", "[PAD]", "[CLS]", "one", "[SEP]"]
    assert encoded[1].attention_mask == [0, 0, 1, 1, 1]
