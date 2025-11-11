import os
from contextlib import contextmanager

import numpy as np
import pytest

from fastembed.rerank.cross_encoder import TextCrossEncoder
from tests.utils import delete_model_cache, should_test_model

CANONICAL_SCORE_VALUES = {
    "Xenova/ms-marco-MiniLM-L-6-v2": np.array([8.500708, -2.541011]),
    "Xenova/ms-marco-MiniLM-L-12-v2": np.array([9.330912, -2.0380247]),
    "BAAI/bge-reranker-base": np.array([6.15733337, -3.65939403]),
    "jinaai/jina-reranker-v1-tiny-en": np.array([2.5911, 0.1122]),
    "jinaai/jina-reranker-v1-turbo-en": np.array([1.8295, -2.8908]),
    "jinaai/jina-reranker-v2-base-multilingual": np.array([1.6533, -1.6455]),
}


MODELS_TO_CACHE = ("Xenova/ms-marco-MiniLM-L-6-v2",)


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
    is_manual = os.getenv("GITHUB_EVENT_NAME") == "workflow_dispatch"

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
            assert np.allclose(
                scores, canonical_scores, atol=1e-3
            ), f"Model: {model_desc.model}, Scores: {scores}, Expected: {canonical_scores}"


@pytest.mark.parametrize("model_name", ["Xenova/ms-marco-MiniLM-L-6-v2"])
def test_batch_rerank(model_cache, model_name: str) -> None:
    with model_cache(model_name) as model:
        query = "What is the capital of France?"
        documents = ["Paris is the capital of France.", "Berlin is the capital of Germany."] * 50
        scores = np.array(list(model.rerank(query, documents, batch_size=10)))

        pairs = [(query, doc) for doc in documents]
        scores2 = np.array(list(model.rerank_pairs(pairs)))
        assert np.allclose(
            scores, scores2, atol=1e-5
        ), f"Model: {model_name}, Scores: {scores}, Scores2: {scores2}"

        canonical_scores = np.tile(CANONICAL_SCORE_VALUES[model_name], 50)

        assert scores.shape == canonical_scores.shape, f"Unexpected shape for model {model_name}"
        assert np.allclose(
            scores, canonical_scores, atol=1e-3
        ), f"Model: {model_name}, Scores: {scores}, Expected: {canonical_scores}"


@pytest.mark.parametrize("model_name", ["Xenova/ms-marco-MiniLM-L-6-v2"])
def test_lazy_load(model_name: str) -> None:
    is_ci = os.getenv("CI")
    model = TextCrossEncoder(model_name=model_name, lazy_load=True)
    assert not hasattr(model.model, "model")
    query = "What is the capital of France?"
    documents = ["Paris is the capital of France.", "Berlin is the capital of Germany."]
    list(model.rerank(query, documents))
    assert hasattr(model.model, "model")

    if is_ci:
        delete_model_cache(model.model._model_dir)


@pytest.mark.parametrize("model_name", ["Xenova/ms-marco-MiniLM-L-6-v2"])
def test_rerank_pairs_parallel(model_cache, model_name: str) -> None:
    with model_cache(model_name) as model:
        query = "What is the capital of France?"
        documents = ["Paris is the capital of France.", "Berlin is the capital of Germany."] * 10
        pairs = [(query, doc) for doc in documents]
        scores_parallel = np.array(list(model.rerank_pairs(pairs, parallel=2, batch_size=10)))
        scores_sequential = np.array(list(model.rerank_pairs(pairs, batch_size=10)))
        assert np.allclose(
            scores_parallel, scores_sequential, atol=1e-5
        ), f"Model: {model_name}, Scores (Parallel): {scores_parallel}, Scores (Sequential): {scores_sequential}"
        canonical_scores = CANONICAL_SCORE_VALUES[model_name]
        assert np.allclose(
            scores_parallel[: len(canonical_scores)], canonical_scores, atol=1e-3
        ), f"Model: {model_name}, Scores (Parallel): {scores_parallel}, Expected: {canonical_scores}"
