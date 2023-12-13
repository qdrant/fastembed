import pytest
import numpy as np
from fastembed.embedding import DefaultEmbedding, Embedding
import timeit
import time

try:
    from sentence_transformers import SentenceTransformer

    ST_IMPORTED = True
except ImportError:
    ST_IMPORTED = False

try:
    from optimum.bettertransformer import BetterTransformer

    OPTIMUM_AVAILABLE = True
except ImportError:
    OPTIMUM_AVAILABLE = False


@pytest.mark.skipif(not ST_IMPORTED, reason="requires sentence-transformers")
@pytest.mark.parametrize("model", Embedding.list_supported_models())
def test_difference_to_sentence_transformers(model):
    """checking if fastembed and sentence-transformers give the same embeddings."""
    model_name_or_path = model["model"]
    if "sentence" in model_name_or_path:
        return
    print("running", model)
    model_fast = DefaultEmbedding(model_name_or_path)
    model_st = SentenceTransformer(model_name_or_path, device="cpu")

    if OPTIMUM_AVAILABLE:
        # from optimum.bettertransformer import BetterTransformer
        model_st._first_module().auto_model.to_bettertransformer()

    sample_sentence = [f"{list(range(i))} " for i in range(64)]

    got = np.stack(list(model_fast.model.onnx_embed(sample_sentence)))
    want = model_st.encode(sample_sentence, normalize_embeddings=True)

    for r, e in zip(got, want):
        cosine_sim = np.dot(r, e) / (np.linalg.norm(e) * np.linalg.norm(r))
        assert cosine_sim > 0.99, f"cosine_sim failed for model {model}. {got} {want}"
    np.testing.assert_almost_equal(got, want, decimal=3)


@pytest.mark.skip
@pytest.mark.parametrize("model", Embedding.list_supported_models()[:1])
def test_performance_vs_sentence_transformers(model):
    """checking claimed cpu performance speedup over torch on cpu."""
    batch_size = 32
    model_name_or_path = model["model"]
    print("running", model)
    model_fast = DefaultEmbedding(model_name_or_path)
    model_st = SentenceTransformer(model_name_or_path, device="cpu")

    if OPTIMUM_AVAILABLE:
        # from optimum.bettertransformer import BetterTransformer
        model_st._first_module().auto_model.to_bettertransformer()

    sample_sentence = [f"{list(range(i))} " for i in range(64)]

    time.sleep(1)
    t1 = timeit.timeit(lambda: list(model_fast.embed(sample_sentence, batch_size=batch_size)), number=10)
    time.sleep(1)
    t2 = timeit.timeit(
        lambda: model_st.encode(sample_sentence, batch_size=batch_size, normalize_embeddings=True), number=10
    )
    assert t1 < t2
