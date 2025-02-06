import os

from PIL import Image
import numpy as np

from fastembed.late_interaction_multimodal import LateInteractionMultimodalEmbedding
from tests.config import TEST_MISC_DIR


# vectors are abridged and rounded for brevity
CANONICAL_IMAGE_VALUES = {
    "Qdrant/colpali-v1.3-fp16": np.array(
        [
            [
                [-0.0345, -0.022, 0.0567, -0.0518, -0.0782, 0.1714, -0.1738],
                [-0.1181, -0.099, 0.0268, 0.0774, 0.0228, 0.0563, -0.1021],
                [-0.117, -0.0683, 0.0371, 0.0921, 0.0107, 0.0659, -0.0666],
                [-0.1393, -0.0948, 0.037, 0.0951, -0.0126, 0.0678, -0.087],
                [-0.0957, -0.081, 0.0404, 0.052, 0.0409, 0.0335, -0.064],
                [-0.0626, -0.0445, 0.056, 0.0592, -0.0229, 0.0409, -0.0301],
                [-0.1299, -0.0691, 0.1097, 0.0728, 0.0123, 0.0519, 0.0122],
            ]
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
}

queries = ["hello world", "flag embedding"]
images = [
    TEST_MISC_DIR / "image.jpeg",
    str(TEST_MISC_DIR / "small_image.jpeg"),
    Image.open((TEST_MISC_DIR / "small_image.jpeg")),
]


def test_batch_embedding():
    is_ci = os.getenv("CI")

    if not is_ci:
        for model_name, expected_result in CANONICAL_IMAGE_VALUES.items():
            print("evaluating", model_name)
            model = LateInteractionMultimodalEmbedding(model_name=model_name)
            result = list(model.embed_image(images, batch_size=2))

            for value in result:
                batch_size, token_num, abridged_dim = expected_result.shape
                assert np.allclose(value[:token_num, :abridged_dim], expected_result, atol=1e-3)


def test_single_embedding():
    is_ci = os.getenv("CI")
    if not is_ci:
        for model_name, expected_result in CANONICAL_IMAGE_VALUES.items():
            print("evaluating", model_name)
            model = LateInteractionMultimodalEmbedding(model_name=model_name)
            result = next(iter(model.embed_image(images, batch_size=6)))
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
