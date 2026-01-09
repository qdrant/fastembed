import os
from contextlib import contextmanager

import pytest
from PIL import Image
import numpy as np

from fastembed import LateInteractionMultimodalEmbedding
from tests.config import TEST_MISC_DIR
from tests.utils import delete_model_cache

# vectors are abridged and rounded for brevity
CANONICAL_IMAGE_VALUES = {
    "Qdrant/colpali-v1.3-fp16": np.array(
        [
            [-0.0345, -0.022, 0.0567, -0.0518, -0.0782, 0.1714, -0.1738],
            [-0.1181, -0.099, 0.0268, 0.0774, 0.0228, 0.0563, -0.1021],
            [-0.117, -0.0683, 0.0371, 0.0921, 0.0107, 0.0659, -0.0666],
            [-0.1393, -0.0948, 0.037, 0.0951, -0.0126, 0.0678, -0.087],
            [-0.0957, -0.081, 0.0404, 0.052, 0.0409, 0.0335, -0.064],
            [-0.0626, -0.0445, 0.056, 0.0592, -0.0229, 0.0409, -0.0301],
            [-0.1299, -0.0691, 0.1097, 0.0728, 0.0123, 0.0519, 0.0122],
        ]
    ),
    "Qdrant/colmodernvbert": np.array(
        [
            [0.11614, -0.15793, -0.11194, 0.0688, 0.08001, 0.10575, -0.07871],
            [0.10094, -0.13301, -0.12069, 0.10932, 0.04645, 0.09884, 0.04048],
            [0.13106, -0.18613, -0.13469, 0.10566, 0.03659, 0.07712, -0.03916],
            [0.09754, -0.09596, -0.04839, 0.14991, 0.05692, 0.10569, -0.08349],
            [0.02576, -0.15651, -0.09977, 0.09707, 0.13412, 0.09994, -0.09931],
            [-0.06741, -0.1787, -0.19677, -0.07618, 0.13102, -0.02131, -0.02437],
            [-0.02776, -0.10187, -0.13793, 0.03835, 0.04766, 0.04701, -0.15635],
        ]
    ),
}

CANONICAL_QUERY_VALUES = {
    "Qdrant/colpali-v1.3-fp16": np.array(
        [
            [-0.0023, 0.1477, 0.1594, 0.046, -0.0196, 0.0554, 0.1567],
            [-0.0139, -0.0057, 0.0932, 0.0052, -0.0678, 0.0131, 0.0537],
            [0.0054, 0.0364, 0.2078, -0.074, 0.0355, 0.061, 0.1593],
            [-0.0076, -0.0154, 0.2266, 0.0103, 0.0089, -0.024, 0.098],
            [-0.0274, 0.0098, 0.2106, -0.0634, 0.0616, -0.0021, 0.0708],
            [0.0074, 0.0025, 0.1631, -0.0802, 0.0418, -0.0219, 0.1022],
            [-0.0165, -0.0106, 0.1672, -0.0768, 0.0389, -0.0038, 0.1137],
        ]
    ),
    "Qdrant/colmodernvbert": np.array(
        [
            [0.05, 0.06557, 0.04026, 0.14981, 0.1842, 0.0263, -0.18706],
            [-0.05664, -0.14028, 0.00649, -0.02849, 0.09034, -0.01494, 0.10693],
            [-0.10147, -0.00716, 0.09084, -0.08236, -0.01849, -0.00972, -0.00461],
            [-0.1233, -0.10814, -0.02337, -0.00329, 0.05984, 0.09934, 0.09846],
            [-0.07053, -0.13119, -0.06487, 0.01508, 0.07459, 0.07655, 0.14821],
            [0.00526, -0.13842, -0.05837, -0.02721, 0.13009, 0.05076, 0.17962],
            [0.00924, -0.14383, -0.03057, -0.03691, 0.11718, 0.037, 0.13344],
        ]
    ),
}

queries = ["hello world", "flag embedding"]
images = [
    TEST_MISC_DIR / "image.jpeg",
    str(TEST_MISC_DIR / "image.jpeg"),
    Image.open((TEST_MISC_DIR / "image.jpeg")),
]

MODELS_TO_CACHE = ("Qdrant/colmodernvbert",)


@pytest.fixture(scope="module")
def model_cache():
    is_ci = os.getenv("CI")
    cache = {}

    @contextmanager
    def get_model(model_name: str):
        lowercase_model_name = model_name.lower()
        if lowercase_model_name not in cache:
            cache[lowercase_model_name] = LateInteractionMultimodalEmbedding(lowercase_model_name)
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


def test_batch_embedding(model_cache):
    for model_name, expected_result in CANONICAL_IMAGE_VALUES.items():
        if model_name.lower() == "Qdrant/colpali-v1.3-fp16".lower() and os.getenv("CI"):
            continue  # colpali is too large for ci

        print("evaluating", model_name)
        with model_cache(model_name) as model:
            result = list(model.embed_image(images, batch_size=2))

            for value in result:
                token_num, abridged_dim = expected_result.shape
                assert np.allclose(value[:token_num, :abridged_dim], expected_result, atol=2e-3)


def test_single_embedding(model_cache):
    for model_name, expected_result in CANONICAL_IMAGE_VALUES.items():
        if model_name.lower() == "Qdrant/colpali-v1.3-fp16".lower() and os.getenv("CI"):
            continue  # colpali is too large for ci
        print("evaluating", model_name)
        with model_cache(model_name) as model:
            result = next(iter(model.embed_image(images, batch_size=6)))
            token_num, abridged_dim = expected_result.shape
            assert np.allclose(result[:token_num, :abridged_dim], expected_result, atol=2e-3)


def test_single_embedding_query(model_cache):
    for model_name, expected_result in CANONICAL_QUERY_VALUES.items():
        if model_name.lower() == "Qdrant/colpali-v1.3-fp16".lower() and os.getenv("CI"):
            continue  # colpali is too large for ci
        print("evaluating", model_name)
        with model_cache(model_name) as model:
            result = next(iter(model.embed_text(queries)))
            token_num, abridged_dim = expected_result.shape
            assert np.allclose(result[:token_num, :abridged_dim], expected_result, atol=2e-3)


def test_get_embedding_size():
    model_name = "Qdrant/colpali-v1.3-fp16"
    assert LateInteractionMultimodalEmbedding.get_embedding_size(model_name) == 128

    model_name = "Qdrant/ColPali-v1.3-fp16"
    assert LateInteractionMultimodalEmbedding.get_embedding_size(model_name) == 128

    model_name = "Qdrant/colmodernvbert"
    assert LateInteractionMultimodalEmbedding.get_embedding_size(model_name) == 128


def test_embedding_size():
    model_name = "Qdrant/colmodernvbert"
    model = LateInteractionMultimodalEmbedding(model_name=model_name, lazy_load=True)
    assert model.embedding_size == 128


def test_token_count(model_cache) -> None:
    model_name = "Qdrant/colmodernvbert"
    with model_cache(model_name) as model:
        documents = ["short doc", "it is a long document to check attention mask for paddings"]
        short_doc_token_count = model.token_count(documents[0])
        long_doc_token_count = model.token_count(documents[1])
        documents_token_count = model.token_count(documents)
        assert short_doc_token_count + long_doc_token_count == documents_token_count
        assert short_doc_token_count + long_doc_token_count == model.token_count(
            documents, batch_size=1
        )
        assert short_doc_token_count + long_doc_token_count < model.token_count(
            documents, include_extension=True
        )
