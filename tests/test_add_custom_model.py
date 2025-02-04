import os
import numpy as np
import pytest

from fastembed.text.text_embedding import TextEmbedding
from tests.utils import delete_model_cache

canonical_vectors = [
    {
        "model": "intfloat/multilingual-e5-small",
        "mean_pooling": True,
        "normalization": True,
        "canonical_vector": [3.1317e-02, 3.0939e-02, -3.5117e-02, -6.7274e-02, 8.5084e-02],
    },
    {
        "model": "intfloat/multilingual-e5-small",
        "mean_pooling": True,
        "normalization": False,
        "canonical_vector": [1.4604e-01, 1.4428e-01, -1.6376e-01, -3.1372e-01, 3.9677e-01],
    },
    {
        "model": "mixedbread-ai/mxbai-embed-xsmall-v1",
        "mean_pooling": False,
        "normalization": False,
        "canonical_vector": [
            2.49407589e-02,
            1.00189969e-02,
            1.07807154e-02,
            3.63860987e-02,
            -2.27128249e-02,
        ],
    },
]

DIMENSIONS = {
    "intfloat/multilingual-e5-small": 384,
    "mixedbread-ai/mxbai-embed-xsmall-v1": 384,
}

SOURCES = {
    "intfloat/multilingual-e5-small": "intfloat/multilingual-e5-small",
    "mixedbread-ai/mxbai-embed-xsmall-v1": "mixedbread-ai/mxbai-embed-xsmall-v1",
}


@pytest.mark.parametrize("scenario", canonical_vectors)
def test_add_custom_model_variations(scenario):
    """
    Tests add_custom_model for different base models and different
    (mean_pooling, normalization) configs. Checks the first 5 dims
    of "hello world" match the scenario's canonical vector.
    """

    is_ci = bool(os.getenv("CI", False))

    base_model_name = scenario["model"]
    mean_pooling = scenario["mean_pooling"]
    normalization = scenario["normalization"]
    cv = np.array(scenario["canonical_vector"], dtype=np.float32)

    suffixes = []
    suffixes.append("mean" if mean_pooling else "no-mean")
    suffixes.append("norm" if normalization else "no-norm")
    suffix_str = "-".join(suffixes)

    custom_model_name = f"{base_model_name}-{suffix_str}"

    dim = DIMENSIONS[base_model_name]
    hf_source = SOURCES[base_model_name]

    model_info = {
        "model": custom_model_name,
        "dim": dim,
        "description": f"{base_model_name} with {suffix_str}",
        "license": "mit",
        "size_in_GB": 0.13,
        "sources": {
            "hf": hf_source,
        },
        "model_file": "onnx/model.onnx",
        "additional_files": [],
    }

    if is_ci and model_info["size_in_GB"] > 1:
        pytest.skip(
            f"Skipping {custom_model_name} on CI due to size_in_GB={model_info['size_in_GB']}"
        )

    TextEmbedding.add_custom_model(
        model_info=model_info, mean_pooling=mean_pooling, normalization=normalization
    )

    model = TextEmbedding(model_name=custom_model_name)

    docs = ["hello world", "flag embedding"]
    embeddings = list(model.embed(docs))
    embeddings = np.stack(embeddings, axis=0)

    assert embeddings.shape == (
        2,
        dim,
    ), f"Expected shape (2, {dim}) for {custom_model_name}, but got {embeddings.shape}"

    num_compare_dims = cv.shape[0]
    assert np.allclose(
        embeddings[0, :num_compare_dims], cv, atol=1e-3
    ), f"Embedding mismatch for {custom_model_name} (first {num_compare_dims} dims)."

    assert not np.allclose(embeddings[0, :], 0.0), "Embedding should not be all zeros."

    if is_ci:
        delete_model_cache(model.model._model_dir)
