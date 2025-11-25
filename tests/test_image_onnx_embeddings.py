import os
from contextlib import contextmanager
from io import BytesIO

import numpy as np
import pytest
import requests
from PIL import Image

from fastembed import ImageEmbedding
from tests.config import TEST_MISC_DIR
from tests.utils import delete_model_cache, should_test_model

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

_MODELS_TO_CACHE = ("Qdrant/clip-ViT-B-32-vision",)
MODELS_TO_CACHE = tuple([x.lower() for x in _MODELS_TO_CACHE])


@pytest.fixture(scope="module")
def model_cache():
    is_ci = os.getenv("CI")
    cache = {}

    @contextmanager
    def get_model(model_name: str):
        lowercase_model_name = model_name.lower()
        if lowercase_model_name not in cache:
            cache[lowercase_model_name] = ImageEmbedding(lowercase_model_name)
        yield cache[lowercase_model_name]
        if lowercase_model_name not in MODELS_TO_CACHE:
            model_inst = cache.pop(lowercase_model_name)
            if is_ci:
                delete_model_cache(model_inst.model._model_dir)
            del model_inst

    yield get_model

    if is_ci:
        for name, model in cache.items():
            delete_model_cache(model.model._model_dir)
    cache.clear()


@pytest.mark.parametrize("model_name", ["Qdrant/clip-ViT-B-32-vision"])
def test_embedding(model_cache, model_name: str) -> None:
    is_ci = os.getenv("CI")
    is_manual = os.getenv("GITHUB_EVENT_NAME") == "workflow_dispatch"

    for model_desc in ImageEmbedding._list_supported_models():
        if not should_test_model(model_desc, model_name, is_ci, is_manual):
            continue

        dim = model_desc.dim

        with model_cache(model_desc.model) as model:
            images = [
                TEST_MISC_DIR / "image.jpeg",
                str(TEST_MISC_DIR / "small_image.jpeg"),
                Image.open((TEST_MISC_DIR / "small_image.jpeg")),
                Image.open(BytesIO(requests.get("https://qdrant.tech/img/logo.png").content)),
            ]
            embeddings = list(model.embed(images))
            embeddings = np.stack(embeddings, axis=0)
            assert embeddings.shape == (len(images), dim)

            canonical_vector = CANONICAL_VECTOR_VALUES[model_desc.model]

            assert np.allclose(
                embeddings[0, : canonical_vector.shape[0]], canonical_vector, atol=1e-3
            ), model_desc.model

            assert np.allclose(embeddings[1], embeddings[2]), model_desc.model


@pytest.mark.parametrize("n_dims,model_name", [(512, "Qdrant/clip-ViT-B-32-vision")])
def test_batch_embedding(model_cache, n_dims: int, model_name: str) -> None:
    with model_cache(model_name) as model:
        n_images = 32
        test_images = [
            TEST_MISC_DIR / "image.jpeg",
            str(TEST_MISC_DIR / "small_image.jpeg"),
            Image.open(TEST_MISC_DIR / "small_image.jpeg"),
        ]
        images = test_images * n_images

        embeddings = list(model.embed(images, batch_size=10))
        embeddings = np.stack(embeddings, axis=0)
        assert np.allclose(embeddings[1], embeddings[2])

        canonical_vector = CANONICAL_VECTOR_VALUES[model_name]

        assert embeddings.shape == (len(test_images) * n_images, n_dims)
        assert np.allclose(embeddings[0, : canonical_vector.shape[0]], canonical_vector, atol=1e-3)


@pytest.mark.parametrize("n_dims,model_name", [(512, "Qdrant/clip-ViT-B-32-vision")])
def test_parallel_processing(model_cache, n_dims: int, model_name: str) -> None:
    with model_cache(model_name) as model:
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


@pytest.mark.parametrize("model_name", ["Qdrant/clip-ViT-B-32-vision"])
def test_lazy_load(model_name: str) -> None:
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


def test_get_embedding_size() -> None:
    assert ImageEmbedding.get_embedding_size(model_name="Qdrant/clip-ViT-B-32-vision") == 512
    assert ImageEmbedding.get_embedding_size(model_name="Qdrant/clip-vit-b-32-vision") == 512


def test_embedding_size() -> None:
    is_ci = os.getenv("CI")
    model_name = "Qdrant/clip-ViT-B-32-vision"
    model = ImageEmbedding(model_name=model_name, lazy_load=True)
    assert model.embedding_size == 512

    model_name = "Qdrant/clip-vit-b-32-vision"
    model = ImageEmbedding(model_name=model_name, lazy_load=True)
    assert model.embedding_size == 512
    if is_ci:
        delete_model_cache(model.model._model_dir)


@pytest.mark.parametrize("model_name", ["Qdrant/clip-ViT-B-32-vision"])
def test_session_options(model_cache, model_name) -> None:
    with model_cache(model_name) as default_model:
        default_session_options = default_model.model.model.get_session_options()
        assert default_session_options.enable_cpu_mem_arena is True
        model = ImageEmbedding(model_name=model_name, enable_cpu_mem_arena=False)
        session_options = model.model.model.get_session_options()
        assert session_options.enable_cpu_mem_arena is False
