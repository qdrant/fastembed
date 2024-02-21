import os

import numpy as np
import pytest

from fastembed.embedding import DefaultEmbedding, JinaEmbedding

CANONICAL_VECTOR_VALUES = {
    "BAAI/bge-small-en": np.array([-0.0232, -0.0255, 0.0174, -0.0639, -0.0006]),
    "BAAI/bge-small-en-v1.5": np.array([0.01522374, -0.02271799, 0.00860278, -0.07424029, 0.00386434]),
    "BAAI/bge-small-zh-v1.5": np.array([-0.01023294, 0.07634465, 0.0691722, -0.04458365, -0.03160762]),
    "BAAI/bge-base-en": np.array([0.0115, 0.0372, 0.0295, 0.0121, 0.0346]),
    "BAAI/bge-base-en-v1.5": np.array([0.01129394, 0.05493144, 0.02615099, 0.00328772, 0.02996045]),
    "BAAI/bge-large-en-v1.5": np.array([0.03434538, 0.03316108, 0.02191251, -0.03713358, -0.01577825]),
    "sentence-transformers/all-MiniLM-L6-v2": np.array([0.0259, 0.0058, 0.0114, 0.0380, -0.0233]),
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": np.array([0.0094,  0.0184,  0.0328,  0.0072, -0.0351]),
    "intfloat/multilingual-e5-large": np.array([0.0098, 0.0045, 0.0066, -0.0354, 0.0070]),
    "xenova/multilingual-e5-large": np.array([0.00975464, 0.00446568, 0.00655449, -0.0354155, 0.00702112]),
    "xenova/paraphrase-multilingual-mpnet-base-v2": np.array(
        [-0.01341097, 0.0416553, -0.00480805, 0.02844842, 0.0505299]
    ),
    "jinaai/jina-embeddings-v2-small-en": np.array([-0.0455, -0.0428, -0.0122, 0.0613, 0.0015]),
    "jinaai/jina-embeddings-v2-base-en": np.array([-0.0332, -0.0509, 0.0287, -0.0043, -0.0077]),
    "nomic-ai/nomic-embed-text-v1": np.array([0.0061, 0.0103, -0.0296, -0.0242, -0.0170]),
}


@pytest.mark.parametrize("embedding_class", [DefaultEmbedding, JinaEmbedding])
def test_embedding(embedding_class):
    is_ubuntu_ci = os.getenv("IS_UBUNTU_CI")

    for model_desc in embedding_class.list_supported_models():
        if is_ubuntu_ci == "false" and model_desc["size_in_GB"] > 1:
            continue

        if model_desc["model"] not in CANONICAL_VECTOR_VALUES:
            continue

        dim = model_desc["dim"]
        model = embedding_class(model_name=model_desc["model"])

        docs = ["hello world", "flag embedding"]
        embeddings = list(model.embed(docs))
        embeddings = np.stack(embeddings, axis=0)
        assert embeddings.shape == (2, dim)

        canonical_vector = CANONICAL_VECTOR_VALUES[model_desc["model"]]
        assert np.allclose(embeddings[0, : canonical_vector.shape[0]], canonical_vector, atol=1e-3), model_desc["model"]


@pytest.mark.parametrize("n_dims,embedding_class", [(384, DefaultEmbedding), (768, JinaEmbedding)])
def test_batch_embedding(n_dims, embedding_class):
    model = embedding_class()

    docs = ["hello world", "flag embedding"] * 100
    embeddings = list(model.embed(docs, batch_size=10))
    embeddings = np.stack(embeddings, axis=0)

    assert embeddings.shape == (200, n_dims)


@pytest.mark.parametrize("n_dims,embedding_class", [(384, DefaultEmbedding), (768, JinaEmbedding)])
def test_parallel_processing(n_dims, embedding_class):
    model = embedding_class()

    docs = ["hello world", "flag embedding"] * 100
    embeddings = list(model.embed(docs, batch_size=10, parallel=2))
    embeddings = np.stack(embeddings, axis=0)

    embeddings_2 = list(model.embed(docs, batch_size=10, parallel=None))
    embeddings_2 = np.stack(embeddings_2, axis=0)

    embeddings_3 = list(model.embed(docs, batch_size=10, parallel=0))
    embeddings_3 = np.stack(embeddings_3, axis=0)

    assert embeddings.shape == (200, n_dims)
    assert np.allclose(embeddings, embeddings_2, atol=1e-3)
    assert np.allclose(embeddings, embeddings_3, atol=1e-3)
