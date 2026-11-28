import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from fastembed.common.model_description import ModelSource
from fastembed.rerank.cross_encoder import TextCrossEncoder
from fastembed.rerank.cross_encoder.custom_text_cross_encoder import CustomTextCrossEncoder
from tests.utils import delete_model_cache, is_manual_run, should_test_model

CANONICAL_SCORE_VALUES = {
    "Xenova/ms-marco-MiniLM-L-6-v2": np.array([8.500708, -2.541011]),
    "Xenova/ms-marco-MiniLM-L-12-v2": np.array([9.330912, -2.0380247]),
    "BAAI/bge-reranker-base": np.array([6.15733337, -3.65939403]),
    "BAAI/bge-reranker-v2-m3": np.array([8.78182220, -5.46485329]),
    "BAAI/bge-reranker-v2-m3-int8": (
        np.array([8.59202957, -5.39362288]),  # arm64, ONNX Runtime 1.27.0
        np.array([8.51154804, -5.56898642]),  # arm64, ONNX Runtime 1.30.0
        np.array([8.85104847, -5.73784494]),  # x86-64 avx2 and vnni, ONNX Runtime 1.30.0
    ),
    "jinaai/jina-reranker-v1-tiny-en": np.array([2.5911, 0.1122]),
    "jinaai/jina-reranker-v1-turbo-en": np.array([1.8295, -2.8908]),
    "jinaai/jina-reranker-v2-base-multilingual": np.array([1.6533, -1.6455]),
}


_MODELS_TO_CACHE = ("Xenova/ms-marco-MiniLM-L-6-v2",)
MODELS_TO_CACHE = tuple([x.lower() for x in _MODELS_TO_CACHE])


@pytest.mark.parametrize("logits", [[0.25, 0.5], [[0.25], [0.5]], [[0.25, 0.75], [0.5, 1.0]]])
def test_custom_reranker_score_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, logits: list[float] | list[list[float]]
) -> None:
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1}, unk_token="[UNK]"))
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        '{"model_max_length": 16, "pad_token": "[PAD]"}', encoding="utf-8"
    )

    session = Mock()
    session.get_inputs.return_value = [SimpleNamespace(name="input_ids")]
    session.run.return_value = [np.array(logits, dtype=np.float32)]
    monkeypatch.setattr("onnxruntime.InferenceSession", Mock(return_value=session))
    monkeypatch.setattr(CustomTextCrossEncoder, "SUPPORTED_MODELS", [])
    model_name = "tests/reranker-score-shapes"
    TextCrossEncoder.add_custom_model(model_name, sources=ModelSource(hf=model_name))
    encoder = TextCrossEncoder(
        model_name, cache_dir=str(tmp_path), specific_model_path=str(tmp_path)
    )

    assert list(encoder.rerank("query", ["first", "second"])) == [0.25, 0.5]
    assert list(encoder.rerank_pairs([("query", "first"), ("query", "second")])) == [0.25, 0.5]


@pytest.fixture(scope="module")
def model_cache():
    is_ci = os.getenv("CI")
    cache = {}

    @contextmanager
    def get_model(model_name: str):
        lowercase_model_name = model_name.lower()
        if lowercase_model_name not in cache:
            cache[lowercase_model_name] = TextCrossEncoder(lowercase_model_name)
        yield cache[lowercase_model_name]
        if lowercase_model_name not in MODELS_TO_CACHE:
            model_inst = cache.pop(lowercase_model_name)
            if is_ci:
                delete_model_cache(model_inst.model._model_dir)
            del model_inst

    yield get_model

    if is_ci:
        for name, model in cache.items():
            delete_model_cache(model.model._model_dir)
    cache.clear()


@pytest.mark.parametrize("model_name", ["Xenova/ms-marco-MiniLM-L-6-v2"])
def test_rerank(model_cache, model_name: str) -> None:
    is_ci = os.getenv("CI")
    is_manual = is_manual_run()

    for model_desc in TextCrossEncoder._list_supported_models():
        if not should_test_model(model_desc, model_name, is_ci, is_manual):
            continue

        with model_cache(model_desc.model) as model:
            query = "What is the capital of France?"
            documents = ["Paris is the capital of France.", "Berlin is the capital of Germany."]
            scores = np.array(list(model.rerank(query, documents)))

            pairs = [(query, doc) for doc in documents]
            scores2 = np.array(list(model.rerank_pairs(pairs)))
            assert np.allclose(
                scores, scores2, atol=1e-5
            ), f"Model: {model_desc.model}, Scores: {scores}, Scores2: {scores2}"

            canonical_scores = CANONICAL_SCORE_VALUES[model_desc.model]
            canonical_variants = (
                canonical_scores if isinstance(canonical_scores, tuple) else (canonical_scores,)
            )
            assert any(
                np.allclose(scores, expected, atol=1e-3) for expected in canonical_variants
            ), f"Model: {model_desc.model}, Scores: {scores}, Expected: {canonical_scores}"


@pytest.mark.parametrize("model_name", ["Xenova/ms-marco-MiniLM-L-6-v2"])
def test_batch_rerank(model_cache, model_name: str) -> None:
    with model_cache(model_name) as model:
        query = "What is the capital of France?"
        # the third document has more tokens, so sorting by length reorders the batches
        unique_documents = [
            "Paris is the capital of France.",
            "Berlin is the capital of Germany.",
            "Madrid, a city of more than three million people, is the capital of Spain.",
        ]
        documents = unique_documents * 50
        scores = np.array(list(model.rerank(query, documents, batch_size=10)))

        pairs = [(query, doc) for doc in documents]
        scores2 = np.array(list(model.rerank_pairs(pairs)))
        assert np.allclose(
            scores, scores2, atol=1e-5
        ), f"Model: {model_name}, Scores: {scores}, Scores2: {scores2}"

        # a single document is a batch of its own, so its score doesn't depend on the sorting
        single_scores = [next(iter(model.rerank(query, [doc]))) for doc in unique_documents]
        expected_scores = np.tile(single_scores, 50)

        assert scores.shape == expected_scores.shape, f"Unexpected shape for model {model_name}"
        assert np.allclose(
            scores, expected_scores, atol=1e-3
        ), f"Model: {model_name}, Scores: {scores}, Expected: {expected_scores}"

        canonical_scores = CANONICAL_SCORE_VALUES[model_name]
        assert np.allclose(
            scores[: len(canonical_scores)], canonical_scores, atol=1e-3
        ), f"Model: {model_name}, Scores: {scores}, Expected: {canonical_scores}"


@pytest.mark.parametrize("model_name", ["Xenova/ms-marco-MiniLM-L-6-v2"])
def test_lazy_load(model_name: str) -> None:
    is_ci = os.getenv("CI")
    model = TextCrossEncoder(model_name=model_name, lazy_load=True)
    assert not hasattr(model.model, "model")
    query = "What is the capital of France?"
    documents = ["Paris is the capital of France.", "Berlin is the capital of Germany."]
    pairs = [(query, doc) for doc in documents]
    assert model.token_count(pairs) > 0
    assert not hasattr(model.model, "model")
    list(model.rerank(query, documents))
    assert hasattr(model.model, "model")

    if is_ci:
        delete_model_cache(model.model._model_dir)


@pytest.mark.parametrize("model_name", ["Xenova/ms-marco-MiniLM-L-6-v2"])
def test_rerank_pairs_parallel(model_cache, model_name: str) -> None:
    with model_cache(model_name) as model:
        query = "What is the capital of France?"
        # the third document has more tokens, so sorting by length reorders the batches
        unique_documents = [
            "Paris is the capital of France.",
            "Berlin is the capital of Germany.",
            "Madrid, a city of more than three million people, is the capital of Spain.",
        ]
        documents = unique_documents * 10
        pairs = [(query, doc) for doc in documents]
        scores_parallel = np.array(list(model.rerank_pairs(pairs, parallel=2, batch_size=10)))
        scores_sequential = np.array(list(model.rerank_pairs(pairs, batch_size=10)))
        assert np.allclose(
            scores_parallel, scores_sequential, atol=1e-5
        ), f"Model: {model_name}, Scores (Parallel): {scores_parallel}, Scores (Sequential): {scores_sequential}"

        # a single pair is a batch of its own, so its score doesn't depend on the sorting
        single_scores = [
            next(iter(model.rerank_pairs([(query, doc)]))) for doc in unique_documents
        ]
        expected_scores = np.tile(single_scores, 10)
        assert np.allclose(
            scores_parallel, expected_scores, atol=1e-3
        ), f"Model: {model_name}, Scores: {scores_parallel}, Expected: {expected_scores}"

        canonical_scores = CANONICAL_SCORE_VALUES[model_name]
        assert np.allclose(
            scores_parallel[: len(canonical_scores)], canonical_scores, atol=1e-3
        ), f"Model: {model_name}, Scores (Parallel): {scores_parallel}, Expected: {canonical_scores}"


@pytest.mark.parametrize("model_name", ["Xenova/ms-marco-MiniLM-L-6-v2"])
def test_token_count(model_cache, model_name: str) -> None:
    with model_cache(model_name) as model:
        pairs = [
            ("What is the capital of France?", "Paris is the capital of France."),
            (
                "Name me a couple of cities were the capitals of Germany?",
                "Berlin is the current capital of Germany, Bonn is a former capital of Germany.",
            ),
        ]
        first_pair_token_count = model.token_count([pairs[0]])
        second_pair_token_count = model.token_count([pairs[1]])
        pairs_token_count = model.token_count(pairs)
        assert first_pair_token_count + second_pair_token_count == pairs_token_count
        assert pairs_token_count == model.token_count(pairs, batch_size=1)


@pytest.mark.parametrize("model_name", ["Xenova/ms-marco-MiniLM-L-6-v2"])
def test_session_options(model_cache, model_name) -> None:
    with model_cache(model_name) as default_model:
        default_session_options = default_model.model.model.get_session_options()
        assert default_session_options.enable_cpu_mem_arena is True
        model = TextCrossEncoder(model_name=model_name, enable_cpu_mem_arena=False)
        session_options = model.model.model.get_session_options()
        assert session_options.enable_cpu_mem_arena is False
