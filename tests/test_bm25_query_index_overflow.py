from pathlib import Path

import mmh3
import numpy as np
import pytest

from fastembed.sparse.bm25 import Bm25


BOUNDARY_TOKEN = "ad1u66pi"
BOUNDARY_TOKEN_ID = 2**31


@pytest.fixture
def model(tmp_path: Path) -> Bm25:
    return Bm25(
        "Qdrant/bm25",
        cache_dir=str(tmp_path),
        specific_model_path=str(tmp_path),
        disable_stemmer=True,
        local_files_only=True,
    )


def test_boundary_token_hash_exceeds_signed_int32_after_absolute_value() -> None:
    assert mmh3.hash(BOUNDARY_TOKEN) == -(2**31)
    assert Bm25.compute_token_id(BOUNDARY_TOKEN) == BOUNDARY_TOKEN_ID


def test_query_embedding_preserves_boundary_token_id(model: Bm25) -> None:
    embedding = list(model.query_embed(BOUNDARY_TOKEN))[0]

    assert embedding.indices.dtype == np.int64
    assert embedding.indices.tolist() == [BOUNDARY_TOKEN_ID]
    assert embedding.values.tolist() == [1]


def test_document_and_query_embeddings_use_same_boundary_token_id(model: Bm25) -> None:
    document_embedding = list(model.embed(BOUNDARY_TOKEN))[0]
    query_embedding = list(model.query_embed(BOUNDARY_TOKEN))[0]

    assert (
        document_embedding.indices.tolist()
        == query_embedding.indices.tolist()
        == [BOUNDARY_TOKEN_ID]
    )


@pytest.mark.parametrize("query", ["AD1U66PI", "(ad1u66pi)!", "ad1u66pi ad1u66pi"])
def test_query_normalization_preserves_boundary_token_id(model: Bm25, query: str) -> None:
    embedding = list(model.query_embed(query))[0]

    assert embedding.indices.tolist() == [BOUNDARY_TOKEN_ID]
    assert embedding.values.tolist() == [1]


@pytest.mark.parametrize("as_generator", [False, True])
def test_query_iterables_handle_mixed_ordinary_and_boundary_tokens(
    model: Bm25, as_generator: bool
) -> None:
    queries = ["hello", BOUNDARY_TOKEN, f"hello {BOUNDARY_TOKEN}", ""]
    query_input = (query for query in queries) if as_generator else queries

    embeddings = list(model.query_embed(query_input))

    hello_id = model.compute_token_id("hello")
    expected_indices = [{hello_id}, {BOUNDARY_TOKEN_ID}, {hello_id, BOUNDARY_TOKEN_ID}, set()]
    assert len(embeddings) == len(expected_indices)
    for embedding, expected in zip(embeddings, expected_indices):
        assert np.issubdtype(embedding.indices.dtype, np.integer)
        assert set(embedding.indices.tolist()) == expected
        assert embedding.values.tolist() == [1] * len(expected)


@pytest.mark.parametrize("query", ["hello", "hello world hello"])
def test_ordinary_query_tokens_have_unit_weights_and_are_deduplicated(
    model: Bm25, query: str
) -> None:
    embedding = list(model.query_embed(query))[0]
    expected_indices = {model.compute_token_id(token) for token in query.split()}

    assert np.issubdtype(embedding.indices.dtype, np.integer)
    assert set(embedding.indices.tolist()) == expected_indices
    assert embedding.values.tolist() == [1] * len(expected_indices)


def test_empty_queries_have_empty_integer_indices_and_values(model: Bm25) -> None:
    embeddings = list(model.query_embed(["", "!!!"]))

    assert len(embeddings) == 2
    for embedding in embeddings:
        assert np.issubdtype(embedding.indices.dtype, np.integer)
        assert embedding.indices.shape == (0,)
        assert embedding.values.shape == (0,)
