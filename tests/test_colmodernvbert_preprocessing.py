"""Offline preprocessing regressions: real transforms/tokenizer, no model weights."""

from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from fastembed.image.transform.operators import Compose
from fastembed.late_interaction_multimodal.colmodernvbert import ColModernVBERT


def expected_prompt(rows: int, cols: int, image_seq_len: int = 2) -> str:
    """Idefics3's row-major local crops followed by the global image."""
    image_tokens = "<image>" * image_seq_len
    fake = "<fake_token_around_image>"
    local_rows = [
        "".join(f"{fake}<row_{row}_col_{col}>{image_tokens}" for col in range(1, cols + 1))
        for row in range(1, rows + 1)
    ]
    local = "\n".join(local_rows) + "\n\n" if local_rows else ""
    image = local + f"{fake}<global-img>{image_tokens}{fake}"
    return f"<|begin_of_text|>User:{image}Describe the image.<end_of_utterance>\nAssistant:"


@pytest.fixture
def model() -> ColModernVBERT:
    # Bypass downloading weights and constructing an ONNX session. Every preprocessing
    # component below is real; only the final inference call is a recording test double.
    model = object.__new__(ColModernVBERT)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    special_tokens = [
        "[UNK]",
        "[PAD]",
        "<image>",
        "<fake_token_around_image>",
        "<global-img>",
        "<|begin_of_text|>",
        "<end_of_utterance>",
    ] + [f"<row_{row}_col_{col}>" for row in range(1, 5) for col in range(1, 5)]
    tokenizer.train_from_iterator(
        [expected_prompt(4, 4)], BpeTrainer(vocab_size=256, special_tokens=special_tokens)
    )
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")
    model.tokenizer = tokenizer
    model.image_seq_len = 2
    model.model = Mock()
    model.model.run.return_value = [np.empty((0,), dtype=np.float32)]
    return model


def configure(model: ColModernVBERT, *, splitting: bool = True, resize: bool = False) -> None:
    model.processor = Compose.from_config(
        {
            "image_processor_type": "Idefics3ImageProcessor",
            "do_resize": resize,
            "size": {"longest_edge": 16},
            "do_image_splitting": splitting,
            "max_image_size": {"longest_edge": 4},
            "do_rescale": True,
            "do_normalize": True,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        }
    )


def assert_prompt(
    model: ColModernVBERT, onnx_input: dict, index: int, grid: tuple[int, int]
) -> None:
    expected = model.tokenizer.encode(expected_prompt(*grid)).ids
    active = onnx_input["attention_mask"][index].astype(bool)
    actual = onnx_input["input_ids"][index][active]
    # Assert actual special-token content as well as exact IDs and row separators.
    actual_tokens = [model.tokenizer.id_to_token(int(token)) for token in actual]
    expected_tokens = [model.tokenizer.id_to_token(token) for token in expected]
    assert actual_tokens == expected_tokens
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("size", "grid"),
    [
        ((4, 8), (2, 1)),
        ((8, 4), (1, 2)),
        ((16, 4), (1, 4)),
        ((4, 16), (4, 1)),
        ((8, 8), (2, 2)),
        ((4, 4), (0, 0)),
        ((5, 9), (3, 2)),
    ],
    ids=["portrait", "landscape", "wide", "tall", "square", "unsplit", "rounded"],
)
def test_image_prompt_retains_patch_grid(model, size, grid) -> None:
    configure(model)
    output = model.onnx_embed_image([Image.new("RGB", size, (32, 64, 128))])
    onnx_input = model.model.run.call_args.args[1]
    assert_prompt(model, onnx_input, 0, grid)
    assert set(onnx_input) == {"pixel_values", "input_ids", "attention_mask"}
    assert output.metadata["image_grid"] == [grid]
    count = grid[0] * grid[1] + 1
    assert output.metadata["patch_counts"] == [count]
    assert onnx_input["pixel_values"].shape == (1, count, 3, 4, 4)


@pytest.mark.parametrize("direction", ["left", "right"])
def test_mixed_image_batch_preserves_grids_and_padding(model, direction) -> None:
    configure(model)
    model.tokenizer.enable_padding(direction=direction, pad_id=1, pad_token="[PAD]")
    sizes = [(4, 8), (8, 4), (16, 4), (8, 8), (4, 4)]
    grids = [(2, 1), (1, 2), (1, 4), (2, 2), (0, 0)]
    output = model.onnx_embed_image([Image.new("RGB", size, (32, 64, 128)) for size in sizes])
    onnx_input = model.model.run.call_args.args[1]
    assert output.metadata["patch_counts"] == [3, 3, 5, 5, 1]
    assert onnx_input["pixel_values"].shape == (5, 5, 3, 4, 4)
    for i, grid in enumerate(grids):
        assert_prompt(model, onnx_input, i, grid)
        count = grid[0] * grid[1] + 1
        assert np.count_nonzero(onnx_input["pixel_values"][i, count:]) == 0
        mask = onnx_input["attention_mask"][i]
        assert np.all(onnx_input["input_ids"][i][mask == 0] == 1)
        padding_count = int(np.sum(mask == 0))
        if padding_count:
            padded = mask[:padding_count] if direction == "left" else mask[-padding_count:]
            assert not np.any(padded)


@pytest.mark.parametrize("size", [(4, 8), (8, 4)])
def test_grid_is_retained_after_longest_edge_resize(model, size) -> None:
    configure(model, resize=True)
    model.onnx_embed_image([Image.new("RGB", size)])
    grid = (4, 2) if size[1] > size[0] else (2, 4)
    assert_prompt(model, model.model.run.call_args.args[1], 0, grid)


def test_disabled_splitting_uses_only_global_prompt(model) -> None:
    configure(model, splitting=False)
    model.onnx_embed_image([Image.new("RGB", (4, 16))])
    onnx_input = model.model.run.call_args.args[1]
    assert_prompt(model, onnx_input, 0, (0, 0))
    assert onnx_input["pixel_values"].shape == (1, 1, 3, 4, 4)


@pytest.mark.parametrize(
    ("grid", "count"),
    [(None, 3), ([], 3), ([(2, 2)], 3), ([(-1, -2)], 3), ([(0, 2)], 1), ([(2, 0)], 1)],
)
def test_missing_or_inconsistent_grid_is_rejected(model, grid, count) -> None:
    inputs = {
        "pixel_values": np.zeros((1, count, 3, 4, 4), dtype=np.float32),
        "attention_mask": np.ones((1, count), dtype=np.int64),
    }
    with pytest.raises(ValueError, match="image patch grid"):
        model._preprocess_onnx_image_input(inputs, image_grid=grid)
