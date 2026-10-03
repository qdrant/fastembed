"""Regression tests for stable SPLADE logarithms, masking, and max pooling."""

import math

import numpy as np
from numpy.typing import NDArray
import pytest

from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.types import NumpyArray
from fastembed.sparse.sparse_embedding_base import SparseEmbedding
from fastembed.sparse.splade_pp import SpladePP


def _post_process(
    logits: NumpyArray, attention_mask: NDArray[np.int64] | None
) -> list[SparseEmbedding]:
    """Run the real SPLADE postprocessor without loading a model or performing inference."""
    model = SpladePP.__new__(SpladePP)
    context = OnnxOutputContext(model_output=logits, attention_mask=attention_mask)
    return list(model._post_process_onnx_output(context))


@pytest.mark.parametrize(
    ("dtype", "value", "relative_tolerance"),
    [
        ("float16", 1e-4, 1e-3),
        ("float32", 1e-8, 5e-7),
        ("float64", 1e-20, 1e-14),
        ("float16", 1e-3, 1e-3),
        ("float32", 1e-6, 5e-7),
        ("float64", 1e-12, 1e-14),
    ],
)
def test_small_positive_weights_are_retained_and_accurate(
    dtype: str, value: float, relative_tolerance: float
) -> None:
    """Retain tiny positive logits and compare weights against a scalar log1p reference."""
    logits = np.array([[[value]]], dtype=dtype)

    embedding = _post_process(logits, np.ones((1, 1), dtype=np.int64))[0]

    assert embedding.indices.tolist() == [0]
    assert embedding.values.shape == (1,)
    expected = math.log1p(float(logits[0, 0, 0]))
    assert embedding.values[0] == pytest.approx(expected, rel=relative_tolerance, abs=0)


@pytest.mark.parametrize(
    ("dtype", "relative_tolerance"),
    [("float16", 1e-3), ("float32", 5e-7), ("float64", 1e-14)],
)
def test_negative_and_zero_logits_remain_filtered(dtype: str, relative_tolerance: float) -> None:
    """Exclude nonpositive logits while retaining ordinary positive vocabulary weights."""
    logits = np.array([[[-1.0, 0.0, 2.0], [-0.25, -0.0, 1.0]]], dtype=dtype)

    embedding = _post_process(logits, np.ones((1, 2), dtype=np.int64))[0]

    assert embedding.indices.tolist() == [2]
    assert embedding.values.tolist() == pytest.approx(
        [math.log1p(2.0)], rel=relative_tolerance, abs=0
    )


@pytest.mark.parametrize(
    ("dtype", "value", "relative_tolerance"),
    [("float16", 1e-4, 1e-3), ("float32", 1e-8, 5e-7), ("float64", 1e-20, 1e-14)],
)
def test_masked_padding_does_not_replace_a_small_valid_weight(
    dtype: str, value: float, relative_tolerance: float
) -> None:
    """Exclude larger padded logits without dropping the valid token's tiny weight."""
    logits = np.array([[[value, 0.0], [10.0, 100.0]]], dtype=dtype)

    embedding = _post_process(logits, np.array([[1, 0]], dtype=np.int64))[0]

    assert embedding.indices.tolist() == [0]
    expected = math.log1p(float(logits[0, 0, 0]))
    assert embedding.values.tolist() == pytest.approx([expected], rel=relative_tolerance, abs=0)


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_all_masked_positions_produce_empty_embeddings(dtype: str) -> None:
    """Return empty sparse arrays when every token position is masked out."""
    logits = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=dtype)

    embedding = _post_process(logits, np.zeros((1, 2), dtype=np.int64))[0]

    assert embedding.indices.shape == (0,)
    assert np.issubdtype(embedding.indices.dtype, np.integer)
    assert embedding.values.shape == (0,)


def test_max_pooling_keeps_batch_order_and_ignores_padding() -> None:
    """Pool only valid positions while preserving each document's place in the batch."""
    logits = np.array(
        [
            [[1, 0, 2, 0], [3, 0, 1, 5], [99, 99, 99, 99]],
            [[0, 7, 0, -2], [0, 4, 0, 0], [100, 100, 100, 100]],
        ],
        dtype=np.float32,
    )

    embeddings = _post_process(logits, np.array([[1, 1, 0], [1, 1, 0]], dtype=np.int64))

    assert len(embeddings) == 2
    assert embeddings[0].indices.tolist() == [0, 2, 3]
    assert embeddings[1].indices.tolist() == [1]
    np.testing.assert_allclose(
        embeddings[0].values, [math.log1p(3), math.log1p(2), math.log1p(5)], rtol=5e-7, atol=0
    )
    np.testing.assert_allclose(embeddings[1].values, [math.log1p(7)], rtol=5e-7, atol=0)


def test_post_processing_preserves_logits_and_attention_mask() -> None:
    """Keep caller-owned logits and attention mask arrays unchanged."""
    logits = np.array([[[-2.0, 1e-8], [1.0, 3.0]]], dtype=np.float32)
    attention_mask = np.array([[1, 0]], dtype=np.int64)
    original_logits = logits.copy()
    original_mask = attention_mask.copy()

    _post_process(logits, attention_mask)

    np.testing.assert_array_equal(logits, original_logits)
    np.testing.assert_array_equal(attention_mask, original_mask)


def test_missing_attention_mask_keeps_existing_error() -> None:
    """Retain the existing document postprocessing error for a missing attention mask."""
    with pytest.raises(ValueError, match="attention_mask must be provided"):
        _post_process(np.zeros((1, 1, 2), dtype=np.float32), None)
