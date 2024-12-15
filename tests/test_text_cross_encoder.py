import os

import numpy as np
import pytest

from fastembed.rerank.cross_encoder import TextCrossEncoder
from tests.utils import delete_model_cache

CANONICAL_SCORE_VALUES = {
    "Xenova/ms-marco-MiniLM-L-6-v2": np.array([8.500708, -2.541011]),
    "Xenova/ms-marco-MiniLM-L-12-v2": np.array([9.330912, -2.0380247]),
    "BAAI/bge-reranker-base": np.array([6.15733337, -3.65939403]),
    "jinaai/jina-reranker-v1-tiny-en": np.array([2.5911, 0.1122]),
    "jinaai/jina-reranker-v1-turbo-en": np.array([1.8295, -2.8908]),
    "jinaai/jina-reranker-v2-base-multilingual": np.array([1.6533, -1.6455]),
}


def test_rerank():
    is_ci = os.getenv("CI")

    for model_desc in TextCrossEncoder.list_supported_models():
        if not is_ci and model_desc["size_in_GB"] > 1:
            continue

        model_name = model_desc["model"]
        model = TextCrossEncoder(model_name=model_name)

        query = "What is the capital of France?"
        documents = ["Paris is the capital of France.", "Berlin is the capital of Germany."]
        scores = np.array(list(model.rerank(query, documents)))

        pairs = [(query, doc)for doc in documents]
        scores2 = np.array(list(model.rerank_pairs(pairs)))
        assert np.allclose(
            scores, scores2, atol=1e-5
        ), f"Model: {model_name}, Scores: {scores}, Scores2: {scores2}"

        canonical_scores = CANONICAL_SCORE_VALUES[model_name]
        assert np.allclose(
            scores, canonical_scores, atol=1e-3
        ), f"Model: {model_name}, Scores: {scores}, Expected: {canonical_scores}"
        if is_ci:
            delete_model_cache(model.model._model_dir)


@pytest.mark.parametrize(
    "model_name",
    [
        model_desc["model"]
        for model_desc in TextCrossEncoder.list_supported_models()
        if model_desc["size_in_GB"] < 1 and model_desc["model"] in CANONICAL_SCORE_VALUES.keys()
    ],
)
def test_batch_rerank(model_name):
    is_ci = os.getenv("CI")

    model = TextCrossEncoder(model_name=model_name)

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
    if is_ci:
        delete_model_cache(model.model._model_dir)


@pytest.mark.parametrize(
    "model_name",
    ["Xenova/ms-marco-MiniLM-L-6-v2"],
)
def test_lazy_load(model_name):
    is_ci = os.getenv("CI")
    model = TextCrossEncoder(model_name=model_name, lazy_load=True)
    assert not hasattr(model.model, "model")
    query = "What is the capital of France?"
    documents = ["Paris is the capital of France.", "Berlin is the capital of Germany."]
    list(model.rerank(query, documents))
    assert hasattr(model.model, "model")

    if is_ci:
        delete_model_cache(model.model._model_dir)

@pytest.mark.parametrize(
    "model_name",
    [
        model_desc["model"]
        for model_desc in TextCrossEncoder.list_supported_models()
        if model_desc["size_in_GB"] < 1 and model_desc["model"] in CANONICAL_SCORE_VALUES.keys()
    ],
)
def test_rerank_pairs_parallel(model_name):
    is_ci = os.getenv("CI")

    model = TextCrossEncoder(model_name=model_name)
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
        scores_parallel[:len(canonical_scores)], canonical_scores, atol=1e-3
    ), f"Model: {model_name}, Scores (Parallel): {scores_parallel}, Expected: {canonical_scores}"
    if is_ci:
        delete_model_cache(model.model._model_dir)
