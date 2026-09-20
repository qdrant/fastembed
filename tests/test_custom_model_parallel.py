"""Public custom-model embedding must preserve serial behavior in worker processes."""

import json
import multiprocessing
import shutil
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType
from fastembed.text import onnx_text_model
from fastembed.text.custom_text_embedding import CustomTextEmbedding

START_METHODS = [
    method
    for method in ("spawn", "forkserver")
    if method in multiprocessing.get_all_start_methods()
]
DOCUMENTS = ["alpha beta", "gamma", "beta alpha gamma", "alpha", "gamma beta", "beta"]
EXPECTED = {
    PoolingType.MEAN: [[2.5, 6.5], [4, 16], [3, 29 / 3], [2, 4], [3.5, 12.5], [3, 9]],
    PoolingType.CLS: [[2, 4], [4, 16], [3, 9], [2, 4], [4, 16], [3, 9]],
    PoolingType.LAST_TOKEN: [[3, 9], [4, 16], [4, 16], [2, 4], [3, 9], [3, 9]],
}


@pytest.fixture
def local_model(tmp_path, monkeypatch):
    monkeypatch.setattr(CustomTextEmbedding, "SUPPORTED_MODELS", [])
    monkeypatch.setattr(CustomTextEmbedding, "POSTPROCESSING_MAPPING", {})
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    fixture = Path(__file__).parent / "misc" / "token_lookup.onnx"
    shutil.copyfile(fixture, tmp_path / "custom-lookup.onnx")
    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "[PAD]": 1, "alpha": 2, "beta": 3, "gamma": 4}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    for filename, metadata in {
        "config.json": {"pad_token_id": 1},
        "tokenizer_config.json": {"model_max_length": 32, "pad_token": "[PAD]"},
        "special_tokens_map.json": {"unk_token": "[UNK]", "pad_token": "[PAD]"},
    }.items():
        (tmp_path / filename).write_text(json.dumps(metadata), encoding="utf-8")
    return tmp_path


def register_model(name, pooling, normalization):
    TextEmbedding.add_custom_model(
        name,
        pooling=pooling,
        normalization=normalization,
        sources=ModelSource(hf="synthetic-unused/local-model"),
        dim=2,
        model_file="custom-lookup.onnx",
    )


def make_model(directory, name, lazy_load):
    return TextEmbedding(
        name,
        cache_dir=str(directory / "cache"),
        specific_model_path=str(directory),
        local_files_only=True,
        providers=["CPUExecutionProvider"],
        threads=1,
        lazy_load=lazy_load,
    )


def expected_vectors(pooling, normalization):
    values = np.array(EXPECTED[pooling], dtype=np.float32)
    if normalization:
        values = values / np.linalg.norm(values, axis=1, keepdims=True)
    return values


@pytest.mark.parametrize("start_method", START_METHODS)
@pytest.mark.parametrize(
    "pooling,normalization,lazy_load,as_generator,batch_size",
    [
        (PoolingType.MEAN, False, True, True, 2),
        (PoolingType.LAST_TOKEN, True, False, False, 1),
        (PoolingType.CLS, False, True, False, 1),
    ],
)
def test_custom_model_parallel_matches_serial(
    local_model,
    monkeypatch,
    start_method,
    pooling,
    normalization,
    lazy_load,
    as_generator,
    batch_size,
):
    # Select a real supported OS process mode; never replace the pool or worker implementation.
    monkeypatch.setattr(onnx_text_model, "get_all_start_methods", lambda: [start_method])
    register_model("Synthetic/Other", PoolingType.CLS, True)
    register_model("Synthetic/Target", pooling, normalization)
    model = make_model(local_model, "synthetic/target", lazy_load)
    expected = expected_vectors(pooling, normalization)
    serial = np.array(list(model.embed(DOCUMENTS, batch_size=batch_size)))
    np.testing.assert_allclose(serial, expected, rtol=1e-6, atol=1e-7)
    documents = (text for text in DOCUMENTS) if as_generator else DOCUMENTS
    parallel = np.array(list(model.embed(documents, batch_size=batch_size, parallel=2)))
    np.testing.assert_allclose(parallel, expected, rtol=1e-6, atol=1e-7)

    # A separate registered configuration must not inherit target pooling/normalization.
    other = make_model(local_model, "Synthetic/Other", True)
    other_parallel = np.array(list(other.embed(DOCUMENTS, batch_size=2, parallel=2)))
    np.testing.assert_allclose(
        other_parallel, expected_vectors(PoolingType.CLS, True), rtol=1e-6, atol=1e-7
    )


@pytest.mark.parametrize("start_method", START_METHODS)
def test_builtin_identity_parallel_control(local_model, monkeypatch, start_method):
    monkeypatch.setattr(onnx_text_model, "get_all_start_methods", lambda: [start_method])
    # The same synthetic graph checks worker infrastructure, not pretrained model quality.
    shutil.copyfile(local_model / "custom-lookup.onnx", local_model / "model_optimized.onnx")
    model = make_model(local_model, "BAAI/bge-small-en-v1.5", True)
    serial = np.array(list(model.embed(DOCUMENTS, batch_size=2)))
    parallel = np.array(list(model.embed(DOCUMENTS, batch_size=2, parallel=2)))
    np.testing.assert_allclose(
        serial, expected_vectors(PoolingType.CLS, True), rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(parallel, serial, rtol=1e-6, atol=1e-7)


def test_custom_small_list_parallel_flag_control(local_model):
    register_model("Synthetic/Small", PoolingType.MEAN, False)
    model = make_model(local_model, "Synthetic/Small", True)
    result = np.array(list(model.embed(DOCUMENTS, batch_size=len(DOCUMENTS) + 1, parallel=2)))
    np.testing.assert_allclose(
        result, expected_vectors(PoolingType.MEAN, False), rtol=1e-6, atol=1e-7
    )


@pytest.mark.parametrize("start_method", START_METHODS)
def test_custom_lazy_model_parallel_first(local_model, monkeypatch, start_method):
    monkeypatch.setattr(onnx_text_model, "get_all_start_methods", lambda: [start_method])
    register_model("Synthetic/LazyFirst", PoolingType.MEAN, True)
    model = make_model(local_model, "synthetic/lazyfirst", True)
    # No serial inference or explicit load precedes the first public embedding call.
    result = np.array(list(model.embed(DOCUMENTS, batch_size=2, parallel=2)))
    np.testing.assert_allclose(
        result, expected_vectors(PoolingType.MEAN, True), rtol=1e-6, atol=1e-7
    )
