import os

import numpy as np
import pytest

from fastembed.text.text_embedding import TextEmbedding

CANONICAL_VECTOR_VALUES = {
    "BAAI/bge-small-en": np.array([-0.0232, -0.0255, 0.0174, -0.0639, -0.0006]),
    "BAAI/bge-small-en-v1.5": np.array(
        [0.01522374, -0.02271799, 0.00860278, -0.07424029, 0.00386434]
    ),
    "BAAI/bge-small-en-v1.5-quantized": np.array(
        [0.01522374, -0.02271799, 0.00860278, -0.07424029, 0.00386434]
    ),
    "BAAI/bge-small-zh-v1.5": np.array(
        [-0.01023294, 0.07634465, 0.0691722, -0.04458365, -0.03160762]
    ),
    "BAAI/bge-base-en": np.array([0.0115, 0.0372, 0.0295, 0.0121, 0.0346]),
    "BAAI/bge-base-en-v1.5": np.array(
        [0.01129394, 0.05493144, 0.02615099, 0.00328772, 0.02996045]
    ),
    "BAAI/bge-large-en-v1.5": np.array(
        [0.03434538, 0.03316108, 0.02191251, -0.03713358, -0.01577825]
    ),
    "BAAI/bge-large-en-v1.5-quantized": np.array(
        [0.03434538, 0.03316108, 0.02191251, -0.03713358, -0.01577825]
    ),
    "sentence-transformers/all-MiniLM-L6-v2": np.array(
        [-0.034478, 0.03102, 0.00673, 0.02611, -0.039362]
    ),
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": np.array(
        [0.0094, 0.0184, 0.0328, 0.0072, -0.0351]
    ),
    "intfloat/multilingual-e5-large": np.array(
        [0.0098, 0.0045, 0.0066, -0.0354, 0.0070]
    ),
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": np.array(
        [-0.01341097, 0.0416553, -0.00480805, 0.02844842, 0.0505299]
    ),
    "jinaai/jina-embeddings-v2-small-en": np.array(
        [-0.0455, -0.0428, -0.0122, 0.0613, 0.0015]
    ),
    "jinaai/jina-embeddings-v2-base-en": np.array(
        [-0.0332, -0.0509, 0.0287, -0.0043, -0.0077]
    ),
    "jinaai/jina-embeddings-v2-base-de": np.array(
        [-0.0085, 0.0417, 0.0342, 0.0309, -0.0149]
    ),
    "nomic-ai/nomic-embed-text-v1": np.array(
        [0.3708 ,  0.2031, -0.3406, -0.2114, -0.3230]
    ),
    "nomic-ai/nomic-embed-text-v1.5": np.array(
        [-0.15407836, -0.03053198, -3.9138033, 0.1910364, 0.13224715]
    ),
    "nomic-ai/nomic-embed-text-v1.5-Q": np.array(
        [-0.12525563,  0.38030425, -3.961622 ,  0.04176439, -0.0758301]
    ),
    "thenlper/gte-large": np.array(
        [-0.01920587, 0.00113156, -0.00708992, -0.00632304, -0.04025577]
    ),
    "mixedbread-ai/mxbai-embed-large-v1": np.array(
        [0.02295546, 0.03196154, 0.016512, -0.04031524, -0.0219634]
    ),
    "snowflake/snowflake-arctic-embed-xs": np.array(
        [0.0092, 0.0619, 0.0196, 0.009, -0.0114]
    ),
    "snowflake/snowflake-arctic-embed-s": np.array(
        [-0.0416, -0.0867, 0.0209, 0.0554, -0.0272]
    ),
    "snowflake/snowflake-arctic-embed-m": np.array(
        [-0.0329, 0.0364, 0.0481, 0.0016, 0.0328]
    ),
    "snowflake/snowflake-arctic-embed-m-long": np.array(
        [0.0080, -0.0266, -0.0335, 0.0282, 0.0143]
    ),
    "snowflake/snowflake-arctic-embed-l": np.array(
        [0.0189, -0.0673, 0.0183, 0.0124, 0.0146]
    ),
    "Qdrant/clip-ViT-B-32-text": np.array([0.0083, 0.0103, -0.0138, 0.0199, -0.0069]),
}


def test_embedding():
    is_ci = os.getenv("CI")

    for model_desc in TextEmbedding.list_supported_models():
        if not is_ci and model_desc["size_in_GB"] > 1:
            continue

        dim = model_desc["dim"]

        model = TextEmbedding(model_name=model_desc["model"])

        docs = ["hello world", "flag embedding"]
        embeddings = list(model.embed(docs))
        embeddings = np.stack(embeddings, axis=0)
        assert embeddings.shape == (2, dim)

        canonical_vector = CANONICAL_VECTOR_VALUES[model_desc["model"]]
        assert np.allclose(
            embeddings[0, : canonical_vector.shape[0]], canonical_vector, atol=1e-3
        ), model_desc["model"]


@pytest.mark.parametrize(
    "n_dims,model_name",
    [(384, "BAAI/bge-small-en-v1.5"), (768, "jinaai/jina-embeddings-v2-base-en")],
)
def test_batch_embedding(n_dims, model_name):
    model = TextEmbedding(model_name=model_name)

    docs = ["hello world", "flag embedding"] * 100
    embeddings = list(model.embed(docs, batch_size=10))
    embeddings = np.stack(embeddings, axis=0)

    assert embeddings.shape == (200, n_dims)


@pytest.mark.parametrize(
    "n_dims,model_name",
    [(384, "BAAI/bge-small-en-v1.5"), (768, "jinaai/jina-embeddings-v2-base-en")],
)
def test_parallel_processing(n_dims, model_name):
    model = TextEmbedding(model_name=model_name)

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
