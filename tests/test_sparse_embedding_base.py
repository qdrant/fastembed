"""Regression tests for sparse embedding index types and dictionary conversion."""

from pathlib import Path

import numpy as np
import pytest

from fastembed.sparse.bm25 import Bm25
from fastembed.sparse.sparse_embedding_base import SparseEmbedding


def test_from_dict_empty_indices_support_numpy_indexing() -> None:
    """Keep empty embeddings usable as NumPy indices without modifying dense values."""
    embedding = SparseEmbedding.from_dict({})

    assert embedding.indices.dtype == np.int64
    assert embedding.indices.shape == (0,)
    assert embedding.values.shape == (0,)
    assert embedding.as_dict() == {}

    dense = np.arange(5, dtype=float)
    expected = dense.copy()
    dense[embedding.indices] = embedding.values

    np.testing.assert_array_equal(dense, expected)


def test_from_dict_preserves_index_value_pairs_and_roundtrip() -> None:
    """Preserve unsorted dictionary entries and their associated weights."""
    data = {3: 1.5, 0: 2.0, 4: -0.25}

    embedding = SparseEmbedding.from_dict(data)

    assert np.issubdtype(embedding.indices.dtype, np.integer)
    np.testing.assert_array_equal(embedding.indices, [3, 0, 4])
    np.testing.assert_array_equal(embedding.values, [1.5, 2.0, -0.25])
    assert embedding.as_dict() == data


@pytest.mark.parametrize("empty_first", [True, False])
def test_concatenating_empty_embedding_preserves_integer_indices(empty_first: bool) -> None:
    """Retain valid integer indices when concatenating an empty embedding in either order."""
    empty = SparseEmbedding.from_dict({})
    nonempty = SparseEmbedding.from_dict({3: 1.5, 0: 2.0, 4: -0.25})
    embeddings = [empty, nonempty] if empty_first else [nonempty, empty]

    indices = np.concatenate([embedding.indices for embedding in embeddings])
    values = np.concatenate([embedding.values for embedding in embeddings])

    assert np.issubdtype(indices.dtype, np.integer)
    dense = np.zeros(5)
    dense[indices] = values

    np.testing.assert_array_equal(dense, [2.0, 0.0, 0.0, 1.5, -0.25])


@pytest.mark.parametrize("document", ["", "the and", "!!!"])
def test_bm25_empty_documents_have_integer_indices(document: str, tmp_path: Path) -> None:
    """Return valid empty sparse indices after BM25 removes every document token."""
    model = Bm25(
        "Qdrant/bm25",
        cache_dir=str(tmp_path),
        specific_model_path=str(tmp_path),
        disable_stemmer=True,
        stopwords={"the", "and"},
        local_files_only=True,
    )

    embeddings = list(model.embed([document]))

    assert len(embeddings) == 1
    embedding = embeddings[0]
    assert embedding.indices.dtype == np.int64
    assert embedding.indices.shape == (0,)
    assert embedding.values.shape == (0,)

    dense = np.zeros(5)
    dense[embedding.indices] = embedding.values

    np.testing.assert_array_equal(dense, np.zeros(5))
