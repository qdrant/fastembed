from pathlib import Path

import numpy as np
import pytest

from fastembed.sparse.bm25 import Bm25
from fastembed.sparse.sparse_text_embedding import SparseTextEmbedding


@pytest.mark.parametrize("model_class", [Bm25, SparseTextEmbedding])
@pytest.mark.parametrize("avg_len", [0.0, -0.0, -1.0, float("nan"), float("inf"), -float("inf")])
def test_bm25_rejects_invalid_average_length(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_class: type[Bm25] | type[SparseTextEmbedding],
    avg_len: float,
) -> None:
    def unexpected_download(*args, **kwargs):
        pytest.fail("Invalid avg_len must be rejected before downloading model files")

    monkeypatch.setattr(Bm25, "download_model", unexpected_download)
    with pytest.raises(ValueError, match="avg_len must be finite and greater than 0"):
        model_class(
            "Qdrant/bm25",
            avg_len=avg_len,
            disable_stemmer=True,
            specific_model_path=str(tmp_path),
            cache_dir=str(tmp_path),
        )


@pytest.mark.parametrize("model_class", [Bm25, SparseTextEmbedding])
@pytest.mark.parametrize(
    "avg_len,expected_weights",
    [
        (0.5, {"fox": 4.4 / 7.7, "dog": 2.2 / 6.7}),
        (3.0, {"fox": 1.375, "dog": 1.0}),
        (256.0, {"fox": 4.4 / 2.310546875, "dog": 2.2 / 1.310546875}),
    ],
)
def test_bm25_positive_average_length_preserves_weights(
    tmp_path: Path,
    model_class: type[Bm25] | type[SparseTextEmbedding],
    avg_len: float,
    expected_weights: dict[str, float],
) -> None:
    model = model_class(
        "Qdrant/bm25",
        avg_len=avg_len,
        disable_stemmer=True,
        specific_model_path=str(tmp_path),
        cache_dir=str(tmp_path),
    )

    embedding = next(iter(model.embed("fox fox dog")))
    assert np.isfinite(embedding.values).all()
    assert dict(zip(embedding.indices, embedding.values)) == pytest.approx(
        {Bm25.compute_token_id(token): weight for token, weight in expected_weights.items()}
    )
    assert next(iter(model.embed(""))).indices.size == 0


def test_bm25_average_length_preserved_in_parallel(tmp_path: Path) -> None:
    model = SparseTextEmbedding(
        "Qdrant/bm25",
        avg_len=64.0,
        disable_stemmer=True,
        specific_model_path=str(tmp_path),
        cache_dir=str(tmp_path),
    )
    documents = ["fox fox dog", "dog", "", "fox"]
    serial = list(model.embed(documents, batch_size=2))
    # More documents than batch_size forces worker processes to be started.
    parallel = list(model.embed(documents, batch_size=2, parallel=2))

    assert len(serial) == len(parallel) == len(documents)
    for expected, actual in zip(serial, parallel):
        assert actual.indices.tolist() == expected.indices.tolist()
        assert actual.values.tolist() == pytest.approx(expected.values.tolist())
