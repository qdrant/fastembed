import os
import numpy as np
import pytest

from fastembed.text.text_embedding import TextEmbedding
from tests.utils import delete_model_cache

# Four variations of the same base model, with different (mean_pooling, normalization).
# Each has a separate canonical vector for "hello world" (first 5 dims).
canonical_vectors = [
    {
        "mean_pooling": True,
        "normalization": True,
        "canonical_vector": [3.1317e-02, 3.0939e-02, -3.5117e-02, -6.7274e-02, 8.5084e-02],
    },
    {
        "mean_pooling": True,
        "normalization": False,
        "canonical_vector": [1.4604e-01, 1.4428e-01, -1.6376e-01, -3.1372e-01, 3.9677e-01],
    },
    {
        "mean_pooling": False,
        "normalization": False,
        "canonical_vector": [1.8612e-01, 9.1158e-02, -1.4521e-01, -3.3533e-01, 3.2876e-01],
    },
    {
        "mean_pooling": False,
        "normalization": True,
        "canonical_vector": [4.6600e-01, 2.1830e-01, -3.3190e-01, -4.2530e-01, 3.3240e-01],
    },
]


@pytest.mark.parametrize("scenario", canonical_vectors)
def test_add_custom_model_variations(scenario):
    """
    Tests that add_custom_model successfully registers the same base model
    with different (mean_pooling, normalization) configurations. We check
    whether we get the correct partial embedding values for "hello world".
    """

    is_ci = os.getenv("CI", False)

    base_model_name = "intfloat/multilingual-e5-small"

    # Build a unique model name to avoid collisions in the registry
    suffix = []
    suffix.append("mean" if scenario["mean_pooling"] else "no-mean")
    suffix.append("norm" if scenario["normalization"] else "no-norm")
    suffix_str = "-".join(suffix)  # e.g. "mean-norm" or "no-mean-norm", etc.

    custom_model_name = f"{base_model_name}-{suffix_str}"

    # Build the base model_info dictionary
    model_info = {
        "model": custom_model_name,  # The registry key
        "dim": 384,
        "description": f"E5-small with {suffix_str}",
        "license": "mit",
        "size_in_GB": 0.13,
        "sources": {
            "hf": "intfloat/multilingual-e5-small",
        },
        "model_file": "onnx/model.onnx",
        "additional_files": [],
    }

    # Possibly skip on CI if the model is large:
    if is_ci and model_info["size_in_GB"] > 1:
        pytest.skip(
            f"Skipping {custom_model_name} on CI because size_in_GB={model_info['size_in_GB']}"
        )

    # Register it so TextEmbedding can find it
    TextEmbedding.add_custom_model(
        model_info=model_info,
        mean_pooling=scenario["mean_pooling"],
        normalization=scenario["normalization"],
    )

    # Instantiate the newly added custom model
    model = TextEmbedding(model_name=custom_model_name)

    # Prepare docs and embed
    docs = ["hello world", "flag embedding"]
    embeddings = list(model.embed(docs))
    embeddings = np.stack(embeddings, axis=0)  # shape => (2, 1024)

    # Check shape
    assert embeddings.shape == (2, model_info["dim"]), (
        f"Expected shape (2, {model_info['dim']}) for {custom_model_name}, "
        f"but got {embeddings.shape}"
    )

    # Compare the first 5 dimensions of the first embedding to the canonical vector
    cv = np.array(scenario["canonical_vector"], dtype=np.float32)  # shape => (5,)
    num_compare_dims = cv.shape[0]
    assert np.allclose(
        embeddings[0, :num_compare_dims], cv, atol=1e-3
    ), f"Embedding mismatch for {custom_model_name} (first {num_compare_dims} dims)."

    # Optional: check that embedding is not all zeros
    assert not np.allclose(embeddings[0, :], 0.0), "Embedding should not be entirely zeros."

    # Clean up cache in CI environment
    if is_ci:
        delete_model_cache(model.model._model_dir)
