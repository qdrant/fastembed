import numpy as np

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
