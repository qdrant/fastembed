import numpy as np
import pytest

from fastembed import (
    TextEmbedding,
    SparseTextEmbedding,
    ImageEmbedding,
    LateInteractionMultimodalEmbedding,
    LateInteractionTextEmbedding,
)
from fastembed.common.utils import iter_batch, last_token_pooling, mean_pooling, normalize


def test_text_list_supported_models():
    for model_type in [
        TextEmbedding,
        SparseTextEmbedding,
        ImageEmbedding,
        LateInteractionMultimodalEmbedding,
        LateInteractionTextEmbedding,
    ]:
        supported_models = model_type.list_supported_models()
        assert isinstance(supported_models, list)
        description = supported_models[0]
        assert isinstance(description, dict)

        assert "model" in description and description["model"]
        if model_type != SparseTextEmbedding:
            assert "dim" in description and description["dim"]
        assert "license" in description and description["license"]
        assert "size_in_GB" in description and description["size_in_GB"]
        assert "model_file" in description and description["model_file"]
        assert "sources" in description and description["sources"]
        assert "hf" in description["sources"] or "url" in description["sources"]


def test_last_token_pooling():
    token_embeddings = np.array(
        [
            [[1.0, 1.0], [2.0, 2.0], [9.0, 9.0], [9.0, 9.0]],  # 2 real tokens, then padding
            [[3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],  # no padding
        ]
    )
    attention_mask = np.array([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=np.int64)

    pooled = last_token_pooling(token_embeddings, attention_mask)

    assert np.allclose(pooled, [[2.0, 2.0], [6.0, 6.0]])


def test_last_token_pooling_with_left_padding():
    token_embeddings = np.array(
        [
            [[9.0, 9.0], [9.0, 9.0], [1.0, 1.0], [2.0, 2.0]],  # padding, then 2 real tokens
            [[3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],  # no padding
        ]
    )
    attention_mask = np.array([[0, 0, 1, 1], [1, 1, 1, 1]], dtype=np.int64)

    pooled = last_token_pooling(token_embeddings, attention_mask)

    assert np.allclose(pooled, [[2.0, 2.0], [6.0, 6.0]])


def test_mean_pooling():
    # 2 real tokens, then padding
    token_embeddings = np.array([[[1.0, 2.0], [3.0, 4.0], [9.0, 9.0]]], dtype=np.float32)
    attention_mask = np.array([[1, 1, 0]], dtype=np.int64)
    # the sum over 8192 tokens of 10.0 exceeds the float16 max of 65504
    long_sequence = np.full((1, 8192, 2), 10.0, dtype=np.float16)

    pooled = mean_pooling(token_embeddings, attention_mask)
    pooled_long_sequence = mean_pooling(long_sequence, np.ones((1, 8192), dtype=np.int64))

    assert pooled.dtype == pooled_long_sequence.dtype == np.float64
    assert np.array_equal(pooled, [[2.0, 3.0]])
    assert np.array_equal(pooled_long_sequence, [[10.0, 10.0]])


def test_normalize_does_not_overflow_float16():
    # the sum of squares, 1024 * 10.0**2, exceeds the float16 max of 65504
    embeddings = np.full((1, 1024), 10.0, dtype=np.float16)

    normalized = normalize(embeddings)

    assert normalized.dtype == np.float16
    assert np.array_equal(normalized, np.full((1, 1024), 1 / 32))


def test_iter_batch_accepts_positive_size():
    assert list(iter_batch([1, 2, 3, 4, 5], 3)) == [[1, 2, 3], [4, 5]]


def test_iter_batch_rejects_non_positive_size():
    with pytest.raises(ValueError, match="batch_size must be >= 1, got 0"):
        iter_batch([1, 2, 3], 0)

    with pytest.raises(ValueError, match="batch_size must be >= 1, got -1"):
        iter_batch([1, 2, 3], -1)
