import numpy as np

from fastembed.late_interaction.late_interaction_text_embedding import LateInteractionTextEmbedding


# vectors are abridged and rounded for brevity
CANONICAL_COLUMN_VALUES = {
    "colbert-ir/colbertv2.0": np.array(
        [
            [0.0759, 0.0841, -0.0299, 0.0374, 0.0254],
            [0.0005, -0.0163, -0.0127, 0.2165, 0.1517],
            [-0.0257, -0.0575, 0.0135, 0.2202, 0.1896],
            [0.0846, 0.0122, 0.0032, -0.0109, -0.1041],
            [0.0477, 0.1078, -0.0314, 0.016, 0.0156],
        ]
    )
}

CANONICAL_QUERY_VALUES = {
    "colbert-ir/colbertv2.0": np.array(
        [
            [0.0824, 0.0872, -0.0324, 0.0418, 0.024],
            [-0.0007, -0.0154, -0.0113, 0.2277, 0.1528],
            [-0.0251, -0.0565, 0.0136, 0.2236, 0.1838],
            [0.0848, 0.0056, 0.0041, -0.0036, -0.1032],
            [0.0574, 0.1072, -0.0332, 0.0233, 0.0209],
            [0.1041, 0.0364, -0.0058, -0.027, -0.0704],
            [0.106, 0.0371, -0.0055, -0.0339, -0.0719],
            [0.1063, 0.0363, 0.0014, -0.0334, -0.0698],
            [0.112, 0.036, 0.0026, -0.0355, -0.0675],
            [0.1184, 0.0441, 0.0166, -0.0169, -0.0244],
            [0.1033, 0.035, 0.0183, 0.0475, 0.0612],
            [-0.0028, -0.014, -0.016, 0.2175, 0.1537],
            [0.0547, 0.0219, -0.007, 0.1748, 0.1154],
            [-0.001, -0.0184, -0.0112, 0.2197, 0.1523],
            [-0.0012, -0.0149, -0.0119, 0.2147, 0.152],
            [-0.0186, -0.0239, -0.014, 0.2196, 0.156],
            [-0.017, -0.0232, -0.0108, 0.2212, 0.157],
            [-0.0109, -0.0024, -0.003, 0.1972, 0.1391],
            [0.0898, 0.0219, -0.0255, 0.0734, -0.0096],
            [0.1143, 0.015, -0.022, 0.0417, -0.0421],
            [0.1056, 0.0091, -0.0137, 0.0129, -0.0619],
            [0.0234, 0.004, -0.0285, 0.1565, 0.0883],
            [-0.0037, -0.0079, -0.0204, 0.1982, 0.1502],
            [0.0988, 0.0377, 0.0226, 0.0309, 0.0508],
            [-0.0103, -0.0128, -0.0035, 0.2114, 0.155],
            [-0.0103, -0.0184, -0.011, 0.2252, 0.157],
            [-0.0033, -0.0292, -0.0097, 0.2237, 0.1607],
            [-0.0198, -0.0257, -0.0193, 0.2265, 0.165],
            [-0.0227, -0.0028, -0.0084, 0.1995, 0.1306],
            [0.0916, 0.0185, -0.0186, 0.0173, -0.0577],
            [0.1022, 0.0228, -0.0174, -0.0102, -0.065],
            [0.1043, 0.0231, -0.0144, -0.0246, -0.067],
        ]
    )
}

docs = ["Hello World"]


def test_batch_embedding():
    docs_to_embed = docs * 10

    for model_name, expected_result in CANONICAL_COLUMN_VALUES.items():
        print("evaluating", model_name)
        model = LateInteractionTextEmbedding(model_name=model_name)
        result = list(model.embed(docs_to_embed, batch_size=6))

        for value in result:
            token_num, abridged_dim = expected_result.shape
            assert np.allclose(value[:, :abridged_dim], expected_result, atol=10e-4)


def test_single_embedding():
    docs_to_embed = docs

    for model_name, expected_result in CANONICAL_COLUMN_VALUES.items():
        print("evaluating", model_name)
        model = LateInteractionTextEmbedding(model_name=model_name, cache_dir="colbert-cache")
        result = next(iter(model.embed(docs_to_embed, batch_size=6)))
        token_num, abridged_dim = expected_result.shape
        assert np.allclose(result[:, :abridged_dim], expected_result, atol=10e-4)


def test_single_embedding_query():
    queries_to_embed = docs

    for model_name, expected_result in CANONICAL_QUERY_VALUES.items():
        print("evaluating", model_name)
        model = LateInteractionTextEmbedding(model_name=model_name, cache_dir="colbert-cache")
        result = next(iter(model.query_embed(queries_to_embed)))
        token_num, abridged_dim = expected_result.shape
        assert np.allclose(result[:, :abridged_dim], expected_result, atol=10e-4)


def test_parallel_processing():
    model = LateInteractionTextEmbedding(
        model_name="colbert-ir/colbertv2.0", cache_dir="colbert-cache"
    )
    token_dim = 128
    docs = ["hello world", "flag embedding"] * 100
    embeddings = list(model.embed(docs, batch_size=10, parallel=2))
    embeddings = np.stack(embeddings, axis=0)

    embeddings_2 = list(model.embed(docs, batch_size=10, parallel=None))
    embeddings_2 = np.stack(embeddings_2, axis=0)

    embeddings_3 = list(model.embed(docs, batch_size=10, parallel=0))
    embeddings_3 = np.stack(embeddings_3, axis=0)

    assert embeddings.shape[0] == len(docs) and embeddings.shape[-1] == token_dim
    assert np.allclose(embeddings, embeddings_2, atol=1e-3)
    assert np.allclose(embeddings, embeddings_3, atol=1e-3)
