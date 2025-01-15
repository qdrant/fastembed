import os

import numpy as np

from fastembed.late_interaction.late_interaction_text_embedding import (
    LateInteractionTextEmbedding,
)
from fastembed.common.pooling import LateInteractionPooler

from tests.utils import delete_model_cache

CANONICAL_COLUMN_VALUES = {
    "colbert-ir/colbertv2.0": np.array(
        [4.0727495e-03, -2.4026826e-03, -6.8204990e-04, -7.1383954e-05, 4.4963313e-03]
    ),
}

docs = ["Hello World"]


def test_batch_embedding():
    is_ci = os.getenv("CI")
    docs_to_embed = docs * 10

    for model_name, expected_result in CANONICAL_COLUMN_VALUES.items():
        print("evaluating", model_name)
        model = LateInteractionTextEmbedding(model_name=model_name)
        pooler = LateInteractionPooler()
        result = list(model.embed(docs_to_embed, batch_size=6))
        pooled_result = pooler.pool(result)
        assert np.allclose(pooled_result[0], expected_result, atol=2e-3)

        if is_ci:
            delete_model_cache(model.model._model_dir)
