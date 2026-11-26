import warnings

import numpy as np
import pytest

from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.late_interaction.colbert import Colbert
from fastembed.late_interaction.lateon import LateOn
from fastembed.late_interaction.token_pooling import ward_groups, pool_tokens_hierarchical


def unit_rows(n: int, dim: int, seed: int = 0, dtype: np.dtype = np.float64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, dim))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(dtype)


# five 1-D points: 1, 2, 10, 11, 20
POINTS = np.array([[1.0], [2.0], [10.0], [11.0], [20.0]])


@pytest.mark.parametrize(
    "n_clusters,expected",
    [
        (5, [[0], [1], [2], [3], [4]]),
        (4, [[0, 1], [2], [3], [4]]),
        (3, [[0, 1], [2, 3], [4]]),
        (2, [[0, 1], [2, 3, 4]]),
        (1, [[0, 1, 2, 3, 4]]),
        (9, [[0], [1], [2], [3], [4]]),  # more clusters than rows: nothing to merge
    ],
)
def test_ward_groups_one_dimensional_points(n_clusters: int, expected: list[list[int]]):
    assert ward_groups(POINTS, n_clusters) == expected


def test_ward_groups_indices_refer_to_input_rows():
    # the same points, shuffled: clusters are reported as indices into the shuffled array
    order = [4, 2, 0, 3, 1]
    groups = ward_groups(POINTS[order], 2)
    assert groups == [[0, 1, 3], [2, 4]]  # {20, 10, 11} and {1, 2}


@pytest.mark.parametrize("n_clusters", [0, -1])
def test_ward_groups_rejects_invalid_n_clusters(n_clusters: int):
    with pytest.raises(ValueError):
        ward_groups(POINTS, n_clusters)


@pytest.mark.parametrize("seed", range(5))
def test_ward_groups_is_a_partition(seed: int):
    x = unit_rows(40, 8, seed)
    groups = ward_groups(1 - x @ x.T, 13)
    assert len(groups) == 13
    flat = [i for g in groups for i in g]
    assert sorted(flat) == list(range(40))  # every row exactly once
    assert all(g == sorted(g) for g in groups)
    assert [g[0] for g in groups] == sorted(g[0] for g in groups)


@pytest.mark.parametrize("seed", range(10))
def test_ward_groups_matches_scipy(seed: int):
    hierarchy = pytest.importorskip("scipy.cluster.hierarchy")
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 120))
    n_clusters = int(rng.integers(1, n))
    x = unit_rows(n, 16, seed)
    observations = 1 - x @ x.T

    with warnings.catch_warnings():
        # SciPy warns that a square distance matrix looks like an uncondensed one, which is
        # expected: the article runs Ward on the rows of this matrix
        warnings.simplefilter("ignore")
        tree = hierarchy.linkage(observations, method="ward", metric="euclidean")
    labels = hierarchy.fcluster(tree, t=n_clusters, criterion="maxclust")
    expected = sorted(
        (np.where(labels == label)[0].tolist() for label in np.unique(labels)),
        key=lambda g: g[0],
    )
    assert ward_groups(observations, n_clusters) == expected


def test_pool_is_noop_without_pooling_factor():
    x = unit_rows(30, 8)
    assert pool_tokens_hierarchical(x, pool_factor=1, protected_tokens=2) is x
    assert pool_tokens_hierarchical(x, pool_factor=0, protected_tokens=2) is x


def test_pool_is_noop_when_too_few_unprotected_tokens():
    x = unit_rows(3, 8)
    # a single unprotected token: nothing to merge
    assert pool_tokens_hierarchical(x, pool_factor=2, protected_tokens=2) is x
    # all tokens protected
    assert pool_tokens_hierarchical(x, pool_factor=2, protected_tokens=3) is x


def test_pool_is_noop_when_target_not_smaller_than_unprotected():
    x = unit_rows(4, 8)
    # n // pool_factor = 2 clusters but only 2 unprotected tokens: no reduction possible
    assert pool_tokens_hierarchical(x, pool_factor=2, protected_tokens=2) is x


@pytest.mark.parametrize("pool_factor", [2, 3, 4])
@pytest.mark.parametrize("protected_tokens", [0, 1, 2])
def test_pool_output_size(pool_factor: int, protected_tokens: int):
    x = unit_rows(60, 8)
    out = pool_tokens_hierarchical(x, pool_factor, protected_tokens)
    assert out.shape[1] == 8
    assert out.shape[0] == 60 // pool_factor + protected_tokens


def test_pool_keeps_protected_tokens_untouched():
    x = unit_rows(40, 8)
    out = pool_tokens_hierarchical(x, pool_factor=2, protected_tokens=2)
    np.testing.assert_array_equal(out[:2], x[:2])


def test_pool_averages_each_cluster():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    cls = np.array([0.0, 0.0, 1.0])
    x = np.stack([cls, a, a, b, b])
    out = pool_tokens_hierarchical(x, pool_factor=2, protected_tokens=1)
    # 5 // 2 = 2 clusters: the two copies of `a`, and the two copies of `b`
    np.testing.assert_allclose(out, np.stack([cls, a, b]))

    near_a = np.array([0.8, 0.6, 0.0])
    x = np.stack([cls, a, near_a, b, b])
    out = pool_tokens_hierarchical(x, pool_factor=2, protected_tokens=1)
    np.testing.assert_allclose(out[1], (a + near_a) / 2)
    np.testing.assert_allclose(out[2], b)


def test_pool_keeps_dtype_and_is_deterministic():
    x = unit_rows(50, 8, dtype=np.float32)
    first = pool_tokens_hierarchical(x, 2, 2)
    second = pool_tokens_hierarchical(x, 2, 2)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, second)


def test_pool_does_not_modify_its_input():
    x = unit_rows(30, 8)
    copy = x.copy()
    pool_tokens_hierarchical(x, 2, 2)
    np.testing.assert_array_equal(x, copy)


# ---- hooks in the model classes, built without loading any model file ----


def make_model(cls):
    model = cls.__new__(cls)
    model.tokenizer = object()  # a loaded tokenizer, so `_ensure_tokenizer` has nothing to do
    model.skip_list = set()
    model.pad_token_id = -1
    return model


def make_output(n_tokens: int = 40, dim: int = 8) -> OnnxOutputContext:
    rng = np.random.default_rng(0)
    return OnnxOutputContext(
        model_output=rng.normal(size=(1, n_tokens, dim)).astype(np.float32),
        attention_mask=np.ones((1, n_tokens), dtype=np.int64),
        input_ids=np.arange(1, n_tokens + 1, dtype=np.int64).reshape(1, n_tokens),
    )


@pytest.mark.parametrize("cls", [Colbert, LateOn])
def test_documents_are_pooled_only_when_requested(cls):
    model = make_model(cls)
    (plain,) = list(model._post_process_onnx_output(make_output(), is_doc=True))
    (pooled,) = list(model._post_process_onnx_output(make_output(), is_doc=True, pool_factor=2))
    (no_pooling,) = list(
        model._post_process_onnx_output(make_output(), is_doc=True, pool_factor=1)
    )

    assert plain.shape == (40, 8)
    np.testing.assert_array_equal(plain, no_pooling)
    assert pooled.shape == (40 // 2 + model.POOLING_PROTECTED_TOKENS, 8)
    np.testing.assert_array_equal(
        pooled[: model.POOLING_PROTECTED_TOKENS], plain[: model.POOLING_PROTECTED_TOKENS]
    )


@pytest.mark.parametrize("cls", [Colbert, LateOn])
def test_queries_are_never_pooled(cls):
    model = make_model(cls)
    (plain,) = list(model._post_process_onnx_output(make_output(), is_doc=False))
    (with_factor,) = list(
        model._post_process_onnx_output(make_output(), is_doc=False, pool_factor=2)
    )
    np.testing.assert_array_equal(plain, with_factor)
    assert plain.shape[0] == 40
