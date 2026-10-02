import numpy as np
import pytest

from fastembed import LateInteractionTextEmbedding
from fastembed.postprocess import Muvera

CANONICAL_VALUES = [-2.61810007e-04, 1.89005750e00, -2.32070747e00]
CANONICAL_QUERY_VALUES = [
    -0.85783903,
    1.1077204,
    -0.09522747,
]  # part of the values are zeros, should be compared with the result of nonzero mask

DIM = 128
K_SIM = 5
DIM_PROJ = 16
R_REPS = 20


def test_single_input():
    model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0", lazy_load=True)
    random_generator = np.random.default_rng(42)
    multivector = random_generator.random((10, 128))

    for muvera in (
        Muvera(dim=DIM, k_sim=K_SIM, dim_proj=DIM_PROJ, r_reps=R_REPS, random_seed=42),
        Muvera.from_multivector_model(model, k_sim=K_SIM, dim_proj=DIM_PROJ, r_reps=R_REPS),
    ):
        fde = muvera.process(multivector)
        assert fde.shape[0] == muvera.embedding_size
        assert np.allclose(fde[:3], CANONICAL_VALUES)

        fde_doc = muvera.process_document(multivector)
        assert fde_doc.shape[0] == muvera.embedding_size
        assert np.allclose(fde, fde_doc)

        fde_query = muvera.process_query(multivector)
        assert fde_query.shape[0] == muvera.embedding_size
        assert np.allclose(fde_query[np.nonzero(fde_query)][:3], CANONICAL_QUERY_VALUES)


def test_empty_multivectors_raise_value_error():
    muvera = Muvera(dim=4, k_sim=2, dim_proj=2, r_reps=3)
    empty = np.empty((0, 4))

    with pytest.raises(ValueError, match="Cannot encode an empty multivector"):
        muvera.process_document(empty)

    with pytest.raises(ValueError, match="Cannot encode an empty multivector"):
        muvera.process_query(empty)


def test_muvera_fills_from_nearest_occupied_cluster():
    muvera = Muvera(dim=2, k_sim=2, dim_proj=2, r_reps=1)
    muvera.simhash_projections[0].get_cluster_ids = lambda vectors: np.array([0, 3])
    vectors = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(
        muvera.process_document(vectors).reshape(4, 2),
        [vectors[0], vectors[0], vectors[0], vectors[1]],
    )


@pytest.mark.parametrize("k_sim", [1, 2, 5, 8])
@pytest.mark.parametrize("assignments", [[0], [0, 0, 0], [0, 1, 0, 1]])
def test_muvera_fills_match_full_matrix_reference(k_sim, assignments):
    from fastembed.postprocess.muvera import hamming_distance_matrix

    n = 2**k_sim
    ids = np.array(assignments) % n
    empty = np.bincount(ids, minlength=n) == 0
    full = hamming_distance_matrix(np.arange(n))
    full[:, empty] = 65
    expected_source_ids = np.argmin(full, axis=1)[empty]
    vectors = np.arange(len(ids), dtype=np.float64)[:, None] + 1
    muvera = Muvera(dim=1, k_sim=k_sim, dim_proj=1, r_reps=1)
    muvera.simhash_projections[0].get_cluster_ids = lambda vectors: ids
    result = muvera.process_document(vectors).reshape(n, 1)
    for empty_id, nearest in zip(np.flatnonzero(empty), expected_source_ids):
        assert result[empty_id, 0] == vectors[np.flatnonzero(ids == nearest)[0], 0]
