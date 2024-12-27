import os

import numpy as np
import pytest

from fastembed.late_interaction_multimodal import LateInteractionMultimodalEmbedding
from tests.utils import delete_model_cache
from tests.config import TEST_MISC_DIR
from PIL import Image

# vectors are abridged and rounded for brevity
CANONICAL_COLUMN_VALUES = {
    "akshayballal/colpali-v1.2-merged": np.array(
        [
            [
                [0.015, 0.051, 0.059, 0.026, -0.061, -0.027, -0.014],
                [-0.22, -0.111, 0.046, 0.081, -0.048, -0.052, -0.086],
                [-0.184, -0.131, 0.004, 0.062, -0.038, -0.059, -0.127],
                [-0.209, -0.113, 0.015, 0.059, -0.035, -0.035, -0.072],
                [-0.031, -0.044, 0.092, -0.005, 0.006, -0.057, -0.061],
                [-0.18, -0.039, 0.031, 0.003, 0.083, -0.041, 0.088],
                [-0.091, 0.023, 0.116, -0.02, 0.039, -0.064, -0.026],
            ]
        ]
    ),
}

CANONICAL_QUERY_VALUES = {
    "akshayballal/colpali-v1.2-merged": np.array(
        [
            [0.158, -0.02, 0.1, -0.023, 0.045, 0.031, 0.071],
            [-0.074, -0.111, 0.065, -0.0, -0.089, -0.003, -0.099],
            [-0.034, -0.014, 0.174, -0.063, -0.09, -0.036, 0.064],
            [-0.07, -0.014, 0.186, -0.013, -0.021, -0.062, 0.107],
            [-0.085, 0.025, 0.179, -0.101, 0.036, -0.089, 0.098],
            [-0.058, 0.031, 0.18, -0.078, 0.023, -0.119, 0.131],
            [-0.067, 0.038, 0.188, -0.079, -0.001, -0.123, 0.127],
            [-0.063, 0.037, 0.204, -0.069, 0.003, -0.118, 0.134],
            [-0.054, 0.036, 0.212, -0.072, -0.001, -0.117, 0.133],
            [-0.044, 0.03, 0.218, -0.077, -0.003, -0.107, 0.139],
            [-0.037, 0.033, 0.22, -0.088, 0.0, -0.095, 0.146],
            [-0.031, 0.041, 0.213, -0.092, 0.001, -0.088, 0.147],
            [-0.026, 0.047, 0.204, -0.089, -0.002, -0.084, 0.144],
            [-0.027, 0.051, 0.199, -0.084, -0.007, -0.083, 0.14],
            [-0.031, 0.056, 0.19, -0.082, -0.011, -0.086, 0.135],
            [-0.008, 0.108, 0.144, -0.095, -0.018, -0.086, 0.085],
        ]
    ),
}

queries = ["hello world", "flag embedding"]
images = [
    TEST_MISC_DIR / "image.jpeg",
    str(TEST_MISC_DIR / "small_image.jpeg"),
    Image.open((TEST_MISC_DIR / "small_image.jpeg")),
]


def test_batch_embedding():
    is_ci = os.getenv("CI")
    docs_to_embed = images

    for model_name, expected_result in CANONICAL_COLUMN_VALUES.items():
        print("evaluating", model_name)
        model = LateInteractionMultimodalEmbedding(model_name=model_name)
        result = list(model.embed_image(docs_to_embed, batch_size=2))

        for value in result:
            batch_size, token_num, abridged_dim = expected_result.shape
            assert np.allclose(value[:token_num, :abridged_dim], expected_result, atol=1e-3)
            break

        if is_ci:
            delete_model_cache(model.model._model_dir)


def test_single_embedding():
    is_ci = os.getenv("CI")
    if not is_ci:
        docs_to_embed = images

        for model_name, expected_result in CANONICAL_COLUMN_VALUES.items():
            print("evaluating", model_name)
            model = LateInteractionMultimodalEmbedding(model_name=model_name)
            result = next(iter(model.embed_images(docs_to_embed, batch_size=6)))
            batch_size, token_num, abridged_dim = expected_result.shape
            assert np.allclose(result[:token_num, :abridged_dim], expected_result, atol=2e-3)


def test_single_embedding_query():
    is_ci = os.getenv("CI")
    if not is_ci:
        queries_to_embed = queries

        for model_name, expected_result in CANONICAL_QUERY_VALUES.items():
            print("evaluating", model_name)
            model = LateInteractionMultimodalEmbedding(model_name=model_name)
            result = next(iter(model.embed_text(queries_to_embed)))
            token_num, abridged_dim = expected_result.shape
            assert np.allclose(result[:token_num, :abridged_dim], expected_result, atol=2e-3)


def test_parallel_processing():
    is_ci = os.getenv("CI")
    if not is_ci:
        model = LateInteractionMultimodalEmbedding(model_name="akshayballal/colpali-v1.2-merged")

        token_dim = 128
        docs = ["hello world", "flag embedding"] * 100
        embeddings = list(model.embed_text(docs, batch_size=10, parallel=2))
        embeddings = np.stack(embeddings, axis=0)

        embeddings_2 = list(model.embed_text(docs, batch_size=10, parallel=None))
        embeddings_2 = np.stack(embeddings_2, axis=0)

        embeddings_3 = list(model.embed_text(docs, batch_size=10, parallel=0))
        embeddings_3 = np.stack(embeddings_3, axis=0)

        assert embeddings.shape[0] == len(docs) and embeddings.shape[-1] == token_dim
        assert np.allclose(embeddings, embeddings_2, atol=1e-3)
        assert np.allclose(embeddings, embeddings_3, atol=1e-3)


@pytest.mark.parametrize(
    "model_name",
    ["akshayballal/colpali-v1.2-merged"],
)
def test_lazy_load(model_name):
    is_ci = os.getenv("CI")
    if not is_ci:
        model = LateInteractionMultimodalEmbedding(model_name=model_name, lazy_load=True)
        assert not hasattr(model.model, "model")

        docs = ["hello world", "flag embedding"]
        list(model.embed_text(docs))
        assert hasattr(model.model, "model")

        model = LateInteractionMultimodalEmbedding(model_name=model_name, lazy_load=True)
        list(model.embed_text(docs))

        model = LateInteractionMultimodalEmbedding(model_name=model_name, lazy_load=True)
        list(model.embed_text(docs))
