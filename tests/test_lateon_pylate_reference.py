from typing import Any, Iterable

import numpy as np
import pytest

from fastembed import LateInteractionTextEmbedding
from fastembed.common.types import NumpyArray


LATEON_MODEL_NAME = "lightonai/LateOn"
REFERENCE_TEXTS = [
    "Hello World",
    "Late interaction models compare token embeddings!",
]
RETRIEVAL_QUERIES = [
    "Which animal purrs?",
    "What is the capital of France?",
]
RETRIEVAL_DOCUMENT_IDS = ["python", "cat", "paris"]
RETRIEVAL_DOCUMENTS = [
    "Python is a programming language used for machine learning.",
    "Cats are small animals that often purr when they are happy.",
    "Paris is the capital and largest city of France.",
]


def _as_numpy_arrays(embeddings: Iterable[NumpyArray]) -> list[np.ndarray]:
    return [np.asarray(embedding, dtype=np.float32) for embedding in embeddings]


def _pylate_model():
    pylate_models = pytest.importorskip(
        "pylate.models", reason="PyLate is required for the LateOn reference test"
    )
    pytest.importorskip("torch", reason="PyLate reference inference requires PyTorch")
    return pylate_models.ColBERT(model_name_or_path=LATEON_MODEL_NAME)


def _pylate_reference_embeddings(texts: list[str], is_query: bool) -> list[np.ndarray]:
    model = _pylate_model()
    embeddings = model.encode(
        texts,
        batch_size=2,
        is_query=is_query,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device="cpu",
    )
    return _as_numpy_arrays(embeddings)


def _rerank(
    queries_embeddings: list[np.ndarray], documents_embeddings: list[np.ndarray]
) -> list[list[Any]]:
    pylate_rank = pytest.importorskip(
        "pylate.rank", reason="PyLate is required for the LateOn retrieval reference test"
    )
    return pylate_rank.rerank(
        documents_ids=[RETRIEVAL_DOCUMENT_IDS] * len(queries_embeddings),
        queries_embeddings=queries_embeddings,
        documents_embeddings=[documents_embeddings] * len(queries_embeddings),
        device="cpu",
    )


def _result_id(result: Any) -> str:
    return result["id"] if isinstance(result, dict) else result.id


def _result_score(result: Any) -> float:
    return result["score"] if isinstance(result, dict) else result.score


@pytest.mark.parametrize("is_query", [False, True])
def test_lateon_matches_pylate_reference(is_query: bool) -> None:
    pylate_embeddings = _pylate_reference_embeddings(REFERENCE_TEXTS, is_query=is_query)

    fastembed_model = LateInteractionTextEmbedding(LATEON_MODEL_NAME, threads=1)
    fastembed_embeddings = _as_numpy_arrays(
        fastembed_model.query_embed(REFERENCE_TEXTS)
        if is_query
        else fastembed_model.embed(REFERENCE_TEXTS, batch_size=2)
    )

    assert len(fastembed_embeddings) == len(pylate_embeddings)
    for fastembed_embedding, pylate_embedding in zip(fastembed_embeddings, pylate_embeddings):
        assert fastembed_embedding.shape == pylate_embedding.shape
        assert np.allclose(fastembed_embedding, pylate_embedding, rtol=1e-3, atol=1e-4)


def test_lateon_retrieval_matches_pylate_reference() -> None:
    pylate_model = _pylate_model()
    pylate_query_embeddings = _as_numpy_arrays(
        pylate_model.encode(
            RETRIEVAL_QUERIES,
            batch_size=2,
            is_query=True,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device="cpu",
        )
    )
    pylate_document_embeddings = _as_numpy_arrays(
        pylate_model.encode(
            RETRIEVAL_DOCUMENTS,
            batch_size=2,
            is_query=False,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device="cpu",
        )
    )

    fastembed_model = LateInteractionTextEmbedding(LATEON_MODEL_NAME, threads=1)
    fastembed_query_embeddings = _as_numpy_arrays(fastembed_model.query_embed(RETRIEVAL_QUERIES))
    fastembed_document_embeddings = _as_numpy_arrays(
        fastembed_model.embed(RETRIEVAL_DOCUMENTS, batch_size=2)
    )

    pylate_results = _rerank(pylate_query_embeddings, pylate_document_embeddings)
    fastembed_results = _rerank(fastembed_query_embeddings, fastembed_document_embeddings)

    assert len(fastembed_results) == len(pylate_results)
    for fastembed_query_results, pylate_query_results in zip(fastembed_results, pylate_results):
        assert [_result_id(result) for result in fastembed_query_results] == [
            _result_id(result) for result in pylate_query_results
        ]
        assert np.allclose(
            [_result_score(result) for result in fastembed_query_results],
            [_result_score(result) for result in pylate_query_results],
            rtol=1e-3,
            atol=1e-3,
        )
