import numpy as np

from fastembed.embedding import DefaultEmbedding, Embedding


CANONICAL_VECTOR_VALUES = {
    "BAAI/bge-small-en": np.array([-0.0232, -0.0255,  0.0174, -0.0639, -0.0006]),
    "BAAI/bge-base-en": np.array([0.0115,  0.0372,  0.0295,  0.0121,  0.0346]),
    "sentence-transformers/all-MiniLM-L6-v2": np.array([0.0259,  0.0058,  0.0114,  0.0380, -0.0233]),
    "intfloat/multilingual-e5-large": np.array([0.0098,  0.0045,  0.0066, -0.0354,  0.0070]),
}


def test_default_embedding():
    for model_desc in Embedding.list_supported_models():
        dim = model_desc["dim"]
        model = DefaultEmbedding(model_name=model_desc["model"])

        docs = ["hello world", "flag embedding"]
        embeddings = list(model.embed(docs))
        embeddings = np.stack(embeddings, axis=0)
        assert embeddings.shape == (2, dim)

        canonical_vector = CANONICAL_VECTOR_VALUES[model_desc["model"]]
        assert np.allclose(embeddings[0, :canonical_vector.shape[0]], canonical_vector, atol=1e-3)


def test_batch_embedding():
    model = DefaultEmbedding()

    docs = ["hello world", "flag embedding"] * 100
    embeddings = list(model.embed(docs, batch_size=10))
    embeddings = np.stack(embeddings, axis=0)

    assert embeddings.shape == (200, 384)
