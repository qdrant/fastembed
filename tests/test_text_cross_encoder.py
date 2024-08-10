import os

import numpy as np
import pytest

from fastembed.rerank.cross_encoder import TextCrossEncoder

CANONICAL_SCORE_VALUES = {
    "cross-encoder/ms-marco-MiniLM-L-6-v2": np.array([8.500708, -2.541011]),
    "cross-encoder/ms-marco-MiniLM-L-12-v2": np.array([9.330912, -2.0380247]),
    "BAAI/bge-reranker-base": np.array([6.1573234, -3.6593902]),
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
        