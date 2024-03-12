from fastembed.sparse.sparse_text_embedding import SparseTextEmbedding

CANONICAL_COLUMN_VALUES = {
    "prithvida/SPLADE_PP_en_v1": [
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
        8484
    ]
}

docs = ["Hello World"]


def test_batch_embedding():
    docs_to_embed = docs * 10

    for model_name, expected_result in CANONICAL_COLUMN_VALUES.items():
        print("evaluating", model_name)
        model = SparseTextEmbedding(model_name=model_name)
        result = next(iter(model.embed(docs_to_embed, batch_size=6)))
        print(result.indices)

        assert sorted(result.indices.tolist()) == expected_result
