import os

import pytest
from PIL import Image
import numpy as np

from fastembed import LateInteractionMultimodalEmbedding
from tests.config import TEST_MISC_DIR


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
            [0.2256, -0.0503, 0.0254, -0.011, -0.0786, 0.2152, -0.0961],
            [-0.0028, -0.0484, -0.0724, -0.0724, -0.0977, 0.0308, -0.0236],
            [0.0035, -0.1075, -0.0877, -0.0207, -0.0828, -0.0294, -0.0253],
            [0.0021, -0.0797, -0.0605, -0.0008, -0.0837, 0.0015, -0.0846],
            [-0.0473, -0.0594, -0.0553, -0.0014, -0.0712, 0.0158, -0.0546],
            [-0.1009, -0.082, -0.0684, -0.1385, -0.0469, -0.0606, -0.0323],
            [-0.0624, 0.006, -0.0498, -0.0127, -0.1115, 0.0076, -0.0888],
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
            [0.05, 0.0656, 0.0403, 0.1498, 0.1842, 0.0263, -0.1871],
            [-0.0566, -0.1403, 0.0065, -0.0285, 0.0903, -0.0149, 0.1069],
            [-0.1015, -0.0072, 0.0908, -0.0824, -0.0185, -0.0097, -0.0046],
            [-0.1233, -0.1081, -0.0234, -0.0033, 0.0598, 0.0993, 0.0985],
            [-0.0705, -0.1312, -0.0649, 0.0151, 0.0746, 0.0765, 0.1482],
            [0.0053, -0.1384, -0.0584, -0.0272, 0.1301, 0.0508, 0.1796],
            [0.0092, -0.1438, -0.0306, -0.0369, 0.1172, 0.037, 0.1334],
        ]
    ),
}

queries = ["hello world", "flag embedding"]
images = [
    TEST_MISC_DIR / "image.jpeg",
    str(TEST_MISC_DIR / "image.jpeg"),
    Image.open((TEST_MISC_DIR / "image.jpeg")),
]


def test_batch_embedding():
    if os.getenv("CI"):
        pytest.skip("Colpali is too large to test in CI")

    for model_name, expected_result in CANONICAL_IMAGE_VALUES.items():
        print("evaluating", model_name)
        model = LateInteractionMultimodalEmbedding(model_name=model_name)
        result = list(model.embed_image(images, batch_size=2))

        for value in result:
            token_num, abridged_dim = expected_result.shape
            assert np.allclose(value[:token_num, :abridged_dim], expected_result, atol=2e-3)


def test_single_embedding():
    if os.getenv("CI"):
        pytest.skip("Colpali is too large to test in CI")

    for model_name, expected_result in CANONICAL_IMAGE_VALUES.items():
        print("evaluating", model_name)
        model = LateInteractionMultimodalEmbedding(model_name=model_name)
        result = next(iter(model.embed_image(images, batch_size=6)))
        token_num, abridged_dim = expected_result.shape
        assert np.allclose(result[:token_num, :abridged_dim], expected_result, atol=2e-3)


def test_single_embedding_query():
    if os.getenv("CI"):
        pytest.skip("Colpali is too large to test in CI")

    for model_name, expected_result in CANONICAL_QUERY_VALUES.items():
        print("evaluating", model_name)
        model = LateInteractionMultimodalEmbedding(model_name=model_name)
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
    if os.getenv("CI"):
        pytest.skip("Colpali is too large to test in CI")
    model_name = "Qdrant/colpali-v1.3-fp16"
    model = LateInteractionMultimodalEmbedding(model_name=model_name, lazy_load=True)
    assert model.embedding_size == 128

    model_name = "Qdrant/ColPali-v1.3-fp16"
    model = LateInteractionMultimodalEmbedding(model_name=model_name, lazy_load=True)
    assert model.embedding_size == 128

    model_name = "Qdrant/colmodernvbert"
    model = LateInteractionMultimodalEmbedding(model_name=model_name, lazy_load=True)
    assert model.embedding_size == 128


def test_token_count() -> None:
    if os.getenv("CI"):
        pytest.skip("Colpali is too large to test in CI")
    model_name = "Qdrant/colpali-v1.3-fp16"
    model = LateInteractionMultimodalEmbedding(model_name=model_name, lazy_load=True)

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
