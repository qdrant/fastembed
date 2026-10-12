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


def test_load_safetensors(tmp_path) -> None:
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
    tensors = load_safetensors(path)
    assert set(tensors) == {"f32", "f16", "bf16", "mapping"}
    for name in ("f32", "f16", "bf16"):
        assert np.array_equal(np.asarray(tensors[name], dtype=np.float32), values), name
    assert tensors["mapping"].tolist() == [2, 0, 1]


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
