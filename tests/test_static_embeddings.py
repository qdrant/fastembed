import json
import os
import warnings
from typing import Iterator

import numpy as np
import pytest

from fastembed.common.safetensors_utils import load_safetensors
from fastembed.text.static_embedding import StaticTextEmbedding
from fastembed.text.text_embedding import TextEmbedding
from tests.test_text_onnx_embeddings import CANONICAL_VECTOR_VALUES
from tests.utils import delete_model_cache, is_manual_run

STATIC_MODELS = [model.model for model in StaticTextEmbedding._list_supported_models()]

CI_MODEL = "minishlab/potion-base-8M"


def _models_to_test() -> list[str]:
    if os.getenv("CI") and not is_manual_run():
        return [CI_MODEL]
    return STATIC_MODELS


@pytest.fixture(scope="module")
def model() -> Iterator[TextEmbedding]:
    model = TextEmbedding(CI_MODEL)
    yield model
    if os.getenv("CI"):
        delete_model_cache(model.model._model_dir)


@pytest.mark.parametrize("model_name", _models_to_test())
def test_canonical_vectors(model_name: str) -> None:
    model = TextEmbedding(model_name)
    assert isinstance(model.model, StaticTextEmbedding)
    embeddings = np.stack(list(model.embed(["hello world", "flag embedding"])))
    assert embeddings.shape == (2, model.embedding_size)
    canonical = CANONICAL_VECTOR_VALUES[model_name]
    assert np.allclose(embeddings[0, : canonical.shape[0]], canonical, atol=1e-3), model_name
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)
    if os.getenv("CI") and model_name != CI_MODEL:
        delete_model_cache(model.model._model_dir)


def test_empty_and_unknown_input_give_zero_vectors(model: TextEmbedding) -> None:
    # Emoji are outside the WordPiece vocabulary, so the second text is only unknown tokens.
    embeddings = np.stack(list(model.embed(["", "🤗🤗", "hello world"])))
    assert not np.isnan(embeddings).any()
    assert np.all(embeddings[0] == 0) and np.all(embeddings[1] == 0)
    assert np.isclose(np.linalg.norm(embeddings[2]), 1.0, atol=1e-5)


def test_unknown_tokens_do_not_change_the_mean(model: TextEmbedding) -> None:
    plain, with_unknown = model.embed(["hello world", "hello 🤗 world"])
    assert np.allclose(plain, with_unknown, atol=1e-6)


def test_batches_match_single_documents(model: TextEmbedding) -> None:
    docs = ["", "a short text", "hello world " * 40, "", "flag embedding", "🤗"] * 7
    batched = np.stack(list(model.embed(docs, batch_size=4)))
    single = np.stack([next(iter(model.embed(doc))) for doc in docs])
    assert np.allclose(batched, single, atol=1e-6)


def test_truncation_to_512_tokens(model: TextEmbedding) -> None:
    long_text = " ".join(f"word{i}" for i in range(2000))
    tokenizer = model.model.tokenizer
    ids = tokenizer.encode(long_text, add_special_tokens=False).ids
    assert len(ids) == 512
    prefix = tokenizer.decode(ids)
    assert np.allclose(
        next(iter(model.embed(long_text))), next(iter(model.embed(prefix))), atol=1e-6
    )


def test_parallel_matches_sequential(model: TextEmbedding) -> None:
    docs = ["hello world", "flag embedding", "", "a much longer document about vectors " * 5] * 100
    sequential = np.stack(list(model.embed(docs, batch_size=10)))
    parallel = np.stack(list(model.embed(docs, batch_size=10, parallel=2)))
    assert np.allclose(sequential, parallel, atol=1e-6)


def test_lazy_load() -> None:
    model = TextEmbedding(CI_MODEL, lazy_load=True)
    assert model.model.embeddings is None
    assert len(list(model.embed(["hello world"]))) == 1
    assert model.model.embeddings is not None


def test_token_count(model: TextEmbedding) -> None:
    assert model.model.token_count("hello world") == 2
    assert model.model.token_count(["hello world", "", "flag embedding"]) == 6
    assert model.model.token_count(" ".join(["word"] * 1000)) == 512


def test_embedding_size() -> None:
    assert TextEmbedding.get_embedding_size("minishlab/potion-retrieval-32M") == 512
    assert TextEmbedding.get_embedding_size("minishlab/potion-base-8M") == 256


def _write_safetensors(path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header, data, offset = {}, b"", 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(raw)],
        }
        data += raw
        offset += len(raw)
    header["__metadata__"] = {"format": "pt"}
    encoded = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(len(encoded).to_bytes(8, "little"))
        f.write(encoded)
        f.write(data)


@pytest.mark.parametrize("mmap", [True, False])
def test_load_safetensors(tmp_path, mmap: bool) -> None:
    values = np.arange(6, dtype=np.float32).reshape(2, 3) / 4
    bf16 = (values.view(np.uint32) >> 16).astype("<u2")
    path = tmp_path / "model.safetensors"
    _write_safetensors(
        path,
        {
            "f32": ("F32", [2, 3], values.astype("<f4").tobytes()),
            "f16": ("F16", [2, 3], values.astype("<f2").tobytes()),
            "bf16": ("BF16", [2, 3], bf16.tobytes()),
            "mapping": ("I64", [3], np.array([2, 0, 1], dtype="<i8").tobytes()),
        },
    )
    tensors = load_safetensors(path, mmap=mmap)
    assert isinstance(tensors["f32"], np.memmap) == mmap
    assert set(tensors) == {"f32", "f16", "bf16", "mapping"}
    for name in ("f32", "f16", "bf16"):
        assert np.array_equal(np.asarray(tensors[name], dtype=np.float32), values), name
    assert tensors["mapping"].tolist() == [2, 0, 1]


@pytest.mark.parametrize("mmap", [True, False])
def test_load_safetensors_empty(tmp_path, mmap: bool) -> None:
    path = tmp_path / "model.safetensors"
    _write_safetensors(
        path,
        {
            "i64": ("I64", [2, 0, 3], b""),
            "f16": ("F16", [0], b""),
            "bool": ("BOOL", [0, 4], b""),
            "bf16": ("BF16", [0, 2], b""),
        },
    )
    tensors = load_safetensors(path, mmap=mmap)
    assert tensors["i64"].dtype == np.int64 and tensors["i64"].shape == (2, 0, 3)
    assert tensors["f16"].dtype == np.float16 and tensors["f16"].shape == (0,)
    assert tensors["bool"].dtype == np.bool_ and tensors["bool"].shape == (0, 4)
    assert tensors["bf16"].dtype == np.float32 and tensors["bf16"].shape == (0, 2)


def test_load_safetensors_unsupported_dtype(tmp_path) -> None:
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, {"empty": ("F8_E4M3", [0], b"")})
    with pytest.raises(ValueError, match="F8_E4M3"):
        load_safetensors(path)


@pytest.mark.parametrize(
    "option",
    [
        {"threads": 2},
        {"providers": ["CPUExecutionProvider"]},
        {"cuda": True},
        {"device_ids": [0]},
        {"enable_cpu_mem_arena": False},
    ],
    ids=lambda option: next(iter(option)),
)
def test_unsupported_options_warn(option: dict) -> None:
    with pytest.warns(UserWarning, match=next(iter(option))):
        TextEmbedding(CI_MODEL, lazy_load=True, **option)


def test_default_options_do_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        TextEmbedding(CI_MODEL, lazy_load=True)


def test_pooling_budget_splits_without_changing_embeddings(model: TextEmbedding) -> None:
    docs = [
        "",
        "hello world",
        "a much longer document about vectors " * 30,
        "🤗",
        "flag embedding",
    ] * 9
    expected = np.stack(list(model.embed(docs)))
    # a budget below one document's tokens halves every batch down to single documents
    tiny_budget = TextEmbedding(CI_MODEL)
    tiny_budget.model.POOLING_CELLS = 5
    assert np.allclose(np.stack(list(tiny_budget.embed(docs))), expected, atol=1e-6)


def test_single_text_matches_batch(model: TextEmbedding) -> None:
    docs = [
        "hello world",
        "",
        "🤗",
        "hello 🤗 world",
        "a much longer document about vectors " * 30,
    ]
    batch = np.stack(list(model.embed(docs, batch_size=len(docs))))
    single = np.stack([next(iter(model.embed(doc))) for doc in docs])
    assert np.allclose(single, batch, atol=1e-6)
    assert single.dtype == np.float32


def test_mmap(model: TextEmbedding) -> None:
    assert not isinstance(model.model.embeddings, np.memmap)  # read into memory by default
    docs = ["hello world", "", "a much longer document about vectors " * 30] * 40
    expected = np.stack(list(model.embed(docs, batch_size=16)))
    mapped = TextEmbedding(CI_MODEL, mmap=True)
    assert isinstance(mapped.model.embeddings, np.memmap)
    assert np.array_equal(np.stack(list(mapped.embed(docs, batch_size=16))), expected)
    parallel = np.stack(list(mapped.embed(docs, batch_size=16, parallel=2)))
    assert np.allclose(parallel, expected, atol=1e-6)


@pytest.mark.skipif(bool(os.getenv("CI")) and not is_manual_run(), reason="large model")
def test_constella_zero_pooling() -> None:
    """Verify model-card pooling: keep special and unknown tokens and weigh counts by sqrt(c)."""
    model = TextEmbedding("Qdrant/constella-zero")
    text = "data data data 🤗 pipeline"
    ids, counts = np.unique(model.model.tokenizer.encode(text).ids, return_counts=True)
    assert 100 in ids and 101 in ids and 102 in ids  # [UNK], [CLS], [SEP]
    # int8 rows, dequantized by their per-row scales in `weights`
    scales = model.model.weights[ids]
    expected = (np.sqrt(counts) * scales).astype(np.float32) @ model.model.embeddings[ids]
    expected /= np.linalg.norm(expected)
    assert np.allclose(next(iter(model.embed(text))), expected, atol=1e-6)
