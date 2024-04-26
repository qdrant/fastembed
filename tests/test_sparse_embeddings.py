import pytest
from fastembed.sparse.sparse_text_embedding import SparseTextEmbedding

CANONICAL_COLUMN_VALUES = {
    "prithvida/Splade_PP_en_v1": {
        "indices": [
            2040,
            2047,
            2088,
            2299,
            2748,
            3011,
            3376,
            3795,
            4774,
            5304,
            5798,
            6160,
            7592,
            7632,
            8484,
        ],
        "values": [
            0.4219532012939453,
            0.4320072531700134,
            2.766580104827881,
            0.3314574658870697,
            1.395172119140625,
            0.021595917642116547,
            0.43770670890808105,
            0.0008370947907678783,
            0.5187209844589233,
            0.17124654352664948,
            0.14742016792297363,
            0.8142819404602051,
            2.803262710571289,
            2.1904349327087402,
            1.0531445741653442,
        ],
    }
}

docs = ["Hello World"]


def test_batch_embedding():
    docs_to_embed = docs * 10

    for model_name, expected_result in CANONICAL_COLUMN_VALUES.items():
        print("evaluating", model_name)
        model = SparseTextEmbedding(model_name=model_name)
        result = next(iter(model.embed(docs_to_embed, batch_size=6)))
        print(result.indices)

        assert result.indices.tolist() == expected_result["indices"]

        for i, value in enumerate(result.values):
            assert pytest.approx(value, abs=0.001) == expected_result["values"][i]


def test_single_embedding():
    docs_to_embed = docs

    for model_name, expected_result in CANONICAL_COLUMN_VALUES.items():
        print("evaluating", model_name)
        model = SparseTextEmbedding(model_name=model_name)
        result = next(iter(model.embed(docs_to_embed, batch_size=6)))
        print(result.indices)

        assert result.indices.tolist() == expected_result["indices"]

        for i, value in enumerate(result.values):
            assert pytest.approx(value, abs=0.001) == expected_result["values"][i]


def test_parallel_processing():
    import numpy as np

    model = SparseTextEmbedding(
        model_name="prithivida/Splade_PP_en_v1",
    )
    docs = ["hello world", "flag embedding"] * 30
    sparse_embeddings_duo = list(model.embed(docs, batch_size=10, parallel=2))
    sparse_embeddings_all = list(model.embed(docs, batch_size=10, parallel=0))
    sparse_embeddings = list(model.embed(docs, batch_size=10, parallel=None))

    assert len(sparse_embeddings) == len(sparse_embeddings_duo) == len(sparse_embeddings_all) == len(docs)

    for sparse_embedding, sparse_embedding_duo, sparse_embedding_all in zip(
        sparse_embeddings, sparse_embeddings_duo, sparse_embeddings_all
    ):
        assert (
            sparse_embedding.indices.tolist()
            == sparse_embedding_duo.indices.tolist()
            == sparse_embedding_all.indices.tolist()
        )
        assert np.allclose(sparse_embedding.values, sparse_embedding_duo.values, atol=1e-3)
        assert np.allclose(sparse_embedding.values, sparse_embedding_all.values, atol=1e-3)
