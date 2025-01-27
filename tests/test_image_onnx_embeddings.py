import os
from io import BytesIO

import numpy as np
import pytest
import requests
from PIL import Image

from fastembed import ImageEmbedding
from tests.config import TEST_MISC_DIR
from tests.utils import delete_model_cache

CANONICAL_VECTOR_VALUES = {
    "Qdrant/clip-ViT-B-32-vision": np.array([-0.0098, 0.0128, -0.0274, 0.002, -0.0059]),
    "Qdrant/resnet50-onnx": np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01046245, 0.01171397, 0.00705971, 0.0]
    ),
    "Qdrant/Unicom-ViT-B-16": np.array(
        [0.0170, -0.0361, 0.0125, -0.0428, -0.0232, 0.0232, -0.0602, -0.0333, 0.0155, 0.0497]
    ),
    "Qdrant/Unicom-ViT-B-32": np.array(
        [0.0418, 0.0550, 0.0003, 0.0253, -0.0185, 0.0016, -0.0368, -0.0402, -0.0891, -0.0186]
    ),
    "jinaai/jina-clip-v1": np.array(
        [-0.029, 0.0216, 0.0396, 0.0283, -0.0023, 0.0151, 0.011, -0.0235, 0.0251, -0.0343]
    ),
}


def test_embedding() -> None:
    is_ci = os.getenv("CI")

    for model_desc in ImageEmbedding.list_supported_models():
        if not is_ci and model_desc["size_in_GB"] > 1:
            continue

        dim = model_desc["dim"]

        model = ImageEmbedding(model_name=model_desc["model"])

        images = [
            TEST_MISC_DIR / "image.jpeg",
            str(TEST_MISC_DIR / "small_image.jpeg"),
            Image.open((TEST_MISC_DIR / "small_image.jpeg")),
            Image.open(BytesIO(requests.get("https://qdrant.tech/img/logo.png").content)),
        ]
        embeddings = list(model.embed(images))
        embeddings = np.stack(embeddings, axis=0)
        assert embeddings.shape == (len(images), dim)

        canonical_vector = CANONICAL_VECTOR_VALUES[model_desc["model"]]

        assert np.allclose(
            embeddings[0, : canonical_vector.shape[0]], canonical_vector, atol=1e-3
        ), model_desc["model"]

        assert np.allclose(embeddings[1], embeddings[2]), model_desc["model"]

        if is_ci:
            delete_model_cache(model.model._model_dir)


@pytest.mark.parametrize("n_dims,model_name", [(512, "Qdrant/clip-ViT-B-32-vision")])
def test_batch_embedding(n_dims, model_name) -> None:
    is_ci = os.getenv("CI")
    model = ImageEmbedding(model_name=model_name)
    n_images = 32
    test_images = [
        TEST_MISC_DIR / "image.jpeg",
        str(TEST_MISC_DIR / "small_image.jpeg"),
        Image.open(TEST_MISC_DIR / "small_image.jpeg"),
    ]
    images = test_images * n_images

    embeddings = list(model.embed(images, batch_size=10))
    embeddings = np.stack(embeddings, axis=0)

    assert embeddings.shape == (len(test_images) * n_images, n_dims)
    if is_ci:
        delete_model_cache(model.model._model_dir)


@pytest.mark.parametrize("n_dims,model_name", [(512, "Qdrant/clip-ViT-B-32-vision")])
def test_parallel_processing(n_dims, model_name) -> None:
    is_ci = os.getenv("CI")
    model = ImageEmbedding(model_name=model_name)

    n_images = 32
    test_images = [
        TEST_MISC_DIR / "image.jpeg",
        str(TEST_MISC_DIR / "small_image.jpeg"),
        Image.open(TEST_MISC_DIR / "small_image.jpeg"),
    ]
    images = test_images * n_images
    embeddings = list(model.embed(images, batch_size=10, parallel=2))
    embeddings = np.stack(embeddings, axis=0)

    embeddings_2 = list(model.embed(images, batch_size=10, parallel=None))
    embeddings_2 = np.stack(embeddings_2, axis=0)

    embeddings_3 = list(model.embed(images, batch_size=10, parallel=0))
    embeddings_3 = np.stack(embeddings_3, axis=0)

    assert embeddings.shape == (n_images * len(test_images), n_dims)
    assert np.allclose(embeddings, embeddings_2, atol=1e-3)
    assert np.allclose(embeddings, embeddings_3, atol=1e-3)
    if is_ci:
        delete_model_cache(model.model._model_dir)


@pytest.mark.parametrize("model_name", ["Qdrant/clip-ViT-B-32-vision"])
def test_lazy_load(model_name) -> None:
    is_ci = os.getenv("CI")
    model = ImageEmbedding(model_name=model_name, lazy_load=True)
    assert not hasattr(model.model, "model")
    images = [
        TEST_MISC_DIR / "image.jpeg",
        str(TEST_MISC_DIR / "small_image.jpeg"),
    ]
    list(model.embed(images))
    assert hasattr(model.model, "model")
    if is_ci:
        delete_model_cache(model.model._model_dir)
