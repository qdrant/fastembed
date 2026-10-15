from typing import Any

import numpy as np
import pytest

from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.late_interaction.colbert import Colbert
from fastembed.late_interaction.jina_colbert import JinaColbert
from fastembed.late_interaction.lateon import LateOn, MLateOn


@pytest.fixture(params=[Colbert, JinaColbert, LateOn, MLateOn])
def model(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Colbert:
    model = object.__new__(request.param)
    model.skip_list = {9}
    model.pad_token_id = 0
    monkeypatch.setattr(model, "_ensure_tokenizer", lambda: None)
    return model


def reference_normalize(vectors: Any) -> Any:
    vectors = vectors.astype(np.float64)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_normalize_finite_document_embeddings(model: Colbert, dtype: Any) -> None:
    vectors = np.full((1, 5, 128), 1e-5, dtype=dtype)
    vectors[0, 0] = 0
    vectors[0, 2] = 100
    vectors[0, 3] = -100
    vectors[0, 4] = np.linspace(-1, 1, 128)
    expected = reference_normalize(vectors[0])
    output = OnnxOutputContext(
        model_output=vectors,
        attention_mask=np.ones((1, 5), dtype=np.int64),
        input_ids=np.ones((1, 5), dtype=np.int64),
    )

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        embeddings = list(model._post_process_onnx_output(output))

    assert len(embeddings) == 1
    assert embeddings[0].dtype == dtype
    assert np.isfinite(embeddings[0]).all()
    np.testing.assert_allclose(embeddings[0], expected, rtol=1e-3, atol=1e-7)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_padding_and_punctuation_are_still_filtered(model: Colbert, dtype: Any) -> None:
    vectors = np.ones((1, 4, 128), dtype=dtype)
    vectors[0, 3] = np.linspace(-1, 1, 128)
    expected = reference_normalize(vectors[0, [0, 3]])
    output = OnnxOutputContext(
        model_output=vectors,
        attention_mask=np.array([[1, 1, 0, 1]], dtype=np.int64),
        input_ids=np.array([[1, 9, 0, 2]], dtype=np.int64),
    )

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        embeddings = list(model._post_process_onnx_output(output))

    assert embeddings[0].dtype == dtype
    np.testing.assert_allclose(embeddings[0], expected, rtol=1e-3, atol=1e-7)


def test_documents_with_no_retained_tokens(model: Colbert) -> None:
    output = OnnxOutputContext(
        model_output=np.ones((1, 2, 128), dtype=np.float16),
        attention_mask=np.array([[0, 1]], dtype=np.int64),
        input_ids=np.array([[0, 9]], dtype=np.int64),
    )

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        embeddings = list(model._post_process_onnx_output(output))

    assert embeddings[0].shape == (0, 128)
    assert embeddings[0].dtype == np.float16


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_query_postprocessing_keeps_existing_semantics(model: Colbert, dtype: Any) -> None:
    vectors = np.full((1, 3, 128), 1e-5, dtype=dtype)
    vectors[0, 0] = 0
    vectors[0, 2] = 100
    expected = reference_normalize(vectors[0]) if isinstance(model, LateOn) else vectors[0].copy()
    output = OnnxOutputContext(
        model_output=vectors,
        attention_mask=np.ones((1, 3), dtype=np.int64),
        input_ids=np.array([[1, 9, 2]], dtype=np.int64),
    )

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        embeddings = list(model._post_process_onnx_output(output, is_doc=False))

    assert embeddings[0].dtype == dtype
    np.testing.assert_allclose(embeddings[0], expected, rtol=1e-3, atol=1e-7)
