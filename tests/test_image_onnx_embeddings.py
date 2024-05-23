import os

import numpy as np
import pytest

from fastembed import ImageEmbedding
from tests.config import TEST_MISC_DIR

CANONICAL_VECTOR_VALUES = {
    "Qdrant/clip-ViT-B-32-vision": np.array([-0.0098, 0.0128, -0.0274, 0.002, -0.0059]),
    "AndrewOgn/resnet_onnx": np.array([0.0322, 0.0027, 0.0144 , 0.0243, 0.0119])
}


def test_embedding():
    is_ci = os.getenv("CI")

    for model_desc in ImageEmbedding.list_supported_models():
        print(model_desc)
        if not is_ci and model_desc["size_in_GB"] > 1:
            continue

        dim = model_desc["dim"]

        model = ImageEmbedding(model_name=model_desc["model"])

        images = [TEST_MISC_DIR / "image.jpeg", str(TEST_MISC_DIR / "small_image.jpeg")]
        embeddings = list(model.embed(images))
        embeddings = np.stack(embeddings, axis=0)
        assert embeddings.shape == (2, dim)

        canonical_vector = CANONICAL_VECTOR_VALUES[model_desc["model"]]
        print(embeddings[0, : canonical_vector.shape[0]])
        assert np.allclose(
            embeddings[0, : canonical_vector.shape[0]], canonical_vector, atol=1e-3
        ), model_desc["model"]


@pytest.mark.parametrize("n_dims,model_name", [(512, "Qdrant/clip-ViT-B-32-vision")])
def test_batch_embedding(n_dims, model_name):
    model = ImageEmbedding(model_name=model_name)
    n_images = 32
    images = [TEST_MISC_DIR / "image.jpeg", str(TEST_MISC_DIR / "small_image.jpeg")] * (
        n_images // 2
    )

    embeddings = list(model.embed(images, batch_size=10))
    embeddings = np.stack(embeddings, axis=0)

    assert embeddings.shape == (n_images, n_dims)


@pytest.mark.parametrize("n_dims,model_name", [(512, "Qdrant/clip-ViT-B-32-vision")])
def test_parallel_processing(n_dims, model_name):
    model = ImageEmbedding(model_name=model_name)

    n_images = 32
    images = [TEST_MISC_DIR / "image.jpeg", str(TEST_MISC_DIR / "small_image.jpeg")] * (
        n_images // 2
    )
    embeddings = list(model.embed(images, batch_size=10, parallel=2))
    embeddings = np.stack(embeddings, axis=0)

    embeddings_2 = list(model.embed(images, batch_size=10, parallel=None))
    embeddings_2 = np.stack(embeddings_2, axis=0)

    embeddings_3 = list(model.embed(images, batch_size=10, parallel=0))
    embeddings_3 = np.stack(embeddings_3, axis=0)

    assert embeddings.shape == (n_images, n_dims)
    assert np.allclose(embeddings, embeddings_2, atol=1e-3)
    assert np.allclose(embeddings, embeddings_3, atol=1e-3)
