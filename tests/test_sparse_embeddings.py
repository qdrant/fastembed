import pytest

from fastembed.text.text_embeddings import TextEmbedding

CANONICAL_COLUMN_VALUES = {
    "prithvida/SPLADE_PP_en_v1": [2040,
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
 8484]
}

docs = ["Hello World"]

@pytest.mark.parametrize(
    "model_name", ["prithvida/SPLADE_PP_en_v1"]
)
def test_batch_embedding(model_name):
    model = TextEmbedding(model_name=model_name)

    docs_to_embed = docs * 10
    binary_vectors = list(model.embed(docs_to_embed, batch_size=10))
    cols = [vec.nonzero()[0].squeeze().tolist() for vec in binary_vectors]
    assert cols[0] == CANONICAL_COLUMN_VALUES[model_name]