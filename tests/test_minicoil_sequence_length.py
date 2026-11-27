from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from fastembed.sparse.minicoil import MiniCOIL
from fastembed.sparse.sparse_text_embedding import SparseTextEmbedding


@pytest.fixture
def local_minicoil(monkeypatch, tmp_path):
    """Use a real tokenizer but avoid downloading model weights for sequence-length tests."""

    def load_tokenizer(model_dir):
        vocab = {"[UNK]": 0, "[PAD]": 1}
        vocab.update({f"word{i}": i + 2 for i in range(12)})
        tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.enable_truncation(max_length=8)
        tokenizer.enable_padding(pad_id=1, pad_token="[PAD]")
        return tokenizer, {}

    monkeypatch.setattr("fastembed.text.onnx_text_model.load_tokenizer", load_tokenizer)
    monkeypatch.setattr(MiniCOIL, "download_model", lambda *args, **kwargs: tmp_path)

    def make_model(max_sequence_length=None):
        kwargs = (
            {} if max_sequence_length is None else {"max_sequence_length": max_sequence_length}
        )
        return SparseTextEmbedding("Qdrant/minicoil-v1", lazy_load=True, **kwargs).model

    return make_model


@pytest.mark.parametrize(
    ("max_sequence_length", "expected_length"),
    [(None, 8), (4, 4), (20, 8)],
)
def test_minicoil_limit_reaches_onnx_input(
    local_minicoil, monkeypatch, max_sequence_length, expected_length
) -> None:
    model = local_minicoil(max_sequence_length)
    model._ensure_tokenizer()
    model.model = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="input_ids")])
    monkeypatch.setattr(
        model,
        "_run_model",
        lambda onnx_input, onnx_output_names=None: np.zeros(
            (*onnx_input["input_ids"].shape, 1), dtype=np.float32
        ),
    )

    text = " ".join(f"word{i}" for i in range(12))
    output = model.onnx_embed([text])

    assert output.input_ids.shape == (1, expected_length)
    assert model.token_count(text) == expected_length


@pytest.mark.parametrize("invalid_length", [0, -1, True, 2.5])
def test_minicoil_rejects_invalid_sequence_limit(local_minicoil, invalid_length) -> None:
    with pytest.raises(ValueError, match="max_sequence_length must be a positive integer"):
        local_minicoil(invalid_length)


@pytest.mark.parametrize("max_sequence_length", [1, 2])
@pytest.mark.parametrize("operation", ["token_count", "embed", "query_embed"])
def test_minicoil_limit_must_leave_room_for_text_and_special_tokens(
    local_minicoil, monkeypatch, max_sequence_length, operation
) -> None:
    tokenizer = Tokenizer(
        models.WordLevel({"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "word": 3}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    tokenizer.enable_truncation(max_length=8)
    monkeypatch.setattr(
        "fastembed.text.onnx_text_model.load_tokenizer",
        lambda model_dir: (tokenizer, {"[CLS]": 1, "[SEP]": 2}),
    )
    model = local_minicoil(max_sequence_length)
    monkeypatch.setattr(
        model,
        "_load_onnx_model",
        lambda **kwargs: pytest.fail("invalid sequence limit must fail before loading ONNX"),
    )

    for _ in range(2):
        with pytest.raises(ValueError, match="max_sequence_length must be at least 3"):
            result = getattr(model, operation)("word word")
            if operation != "token_count":
                list(result)


def test_minicoil_parallel_workers_receive_sequence_limit(local_minicoil, monkeypatch) -> None:
    model = local_minicoil(4)
    captured = {}

    class RecordingPool:
        def __init__(self, **kwargs):
            pass

        def ordered_map(self, batches, **kwargs):
            captured.update(kwargs)
            return iter(())

    monkeypatch.setattr("fastembed.text.onnx_text_model.ParallelWorkerPool", RecordingPool)

    assert list(model.embed(["word0", "word1"], batch_size=1, parallel=2)) == []
    assert captured["max_sequence_length"] == 4
