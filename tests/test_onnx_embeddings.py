import numpy as np

from fastembed.embedding import DefaultEmbedding


def test_default_embedding():
    model = DefaultEmbedding()

    docs = ["hello world", "flag embedding"]
    embeddings = np.array(model.embed(docs))
    assert embeddings.shape == (2, 384)

test_default_embedding()