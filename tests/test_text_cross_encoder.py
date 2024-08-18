import os

import numpy as np
import pytest

from fastembed.rerank.cross_encoder import TextCrossEncoder

CANONICAL_SCORE_VALUES = {
    "Xenova/ms-marco-MiniLM-L-6-v2": np.array([8.500708, -2.541011]),
    "Xenova/ms-marco-MiniLM-L-12-v2": np.array([9.330912, -2.0380247]),
    "BAAI/bge-reranker-base": np.array([6.72338772, -2.28010106]),
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
        scores = np.array(model.rerank(query, documents))

        canonical_scores = CANONICAL_SCORE_VALUES[model_name]
        assert np.allclose(
            scores, canonical_scores, atol=1e-3
        ), f"Model: {model_name}, Scores: {scores}, Expected: {canonical_scores}"


@pytest.mark.parametrize(
    "model_name",
    ["Xenova/ms-marco-MiniLM-L-6-v2", "Xenova/ms-marco-MiniLM-L-12-v2", "BAAI/bge-reranker-base"],
)
def test_batch_rerank(model_name):
    model = TextCrossEncoder(model_name=model_name)

    query = "What is the capital of France?"
    documents = ["Paris is the capital of France.", "Berlin is the capital of Germany."] * 50
    scores = np.array(model.rerank(query, documents, batch_size=10))

    canonical_scores = np.tile(CANONICAL_SCORE_VALUES[model_name], 50)

    assert scores.shape == canonical_scores.shape, f"Unexpected shape for model {model_name}"
    assert np.allclose(
        scores, canonical_scores, atol=1e-3
    ), f"Model: {model_name}, Scores: {scores}, Expected: {canonical_scores}"
