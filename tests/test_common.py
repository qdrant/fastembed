from pathlib import Path

import numpy as np

from fastembed import (
    TextEmbedding,
    SparseTextEmbedding,
    ImageEmbedding,
    LateInteractionMultimodalEmbedding,
    LateInteractionTextEmbedding,
)
from fastembed.common.model_management import ModelManagement
from fastembed.common.utils import last_token_pooling
from fastembed.sparse.bm25 import Bm25, supported_bm25_models


def test_text_list_supported_models():
    for model_type in [
        TextEmbedding,
        SparseTextEmbedding,
        ImageEmbedding,
        LateInteractionMultimodalEmbedding,
        LateInteractionTextEmbedding,
    ]:
        supported_models = model_type.list_supported_models()
        assert isinstance(supported_models, list)
        description = supported_models[0]
        assert isinstance(description, dict)

        assert "model" in description and description["model"]
        if model_type != SparseTextEmbedding:
            assert "dim" in description and description["dim"]
        assert "license" in description and description["license"]
        assert "size_in_GB" in description and description["size_in_GB"]
        assert "model_file" in description and description["model_file"]
        assert "sources" in description and description["sources"]
        assert "hf" in description["sources"] or "url" in description["sources"]


def test_last_token_pooling():
    token_embeddings = np.array(
        [
            [[1.0, 1.0], [2.0, 2.0], [9.0, 9.0], [9.0, 9.0]],  # 2 real tokens, then padding
            [[3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],  # no padding
        ]
    )
    attention_mask = np.array([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=np.int64)

    pooled = last_token_pooling(token_embeddings, attention_mask)

    assert np.allclose(pooled, [[2.0, 2.0], [6.0, 6.0]])


def test_last_token_pooling_with_left_padding():
    token_embeddings = np.array(
        [
            [[9.0, 9.0], [9.0, 9.0], [1.0, 1.0], [2.0, 2.0]],  # padding, then 2 real tokens
            [[3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],  # no padding
        ]
    )
    attention_mask = np.array([[0, 0, 1, 1], [1, 1, 1, 1]], dtype=np.int64)

    pooled = last_token_pooling(token_embeddings, attention_mask)

    assert np.allclose(pooled, [[2.0, 2.0], [6.0, 6.0]])


def test_bm25_resolves_offline_without_network_fetch(tmp_path, monkeypatch):
    """A mock model (``Qdrant/bm25``) whose real required files are already cached
    must short-circuit locally and never trigger a network fetch.

    ``bm25`` uses ``model_file="mock.file"`` as a placeholder that never exists on
    disk; its real required files are ``additional_files`` (the ``{lang}.txt``
    stop-word lists). The offline cache probe must therefore ignore the mock file.
    """
    model = supported_bm25_models[0]
    assert model.model_file == ModelManagement.MOCK_MODEL_FILE

    hf_repo = model.sources.hf
    snapshot_dir = tmp_path / f"models--{hf_repo.replace('/', '--')}"
    snapshot_dir.mkdir(parents=True)
    # Seed the cache with the *real* required files only (NOT the mock model_file).
    for fname in model.additional_files:
        (snapshot_dir / fname).write_text("stopword\n")

    seen_local_files_only: list[bool] = []

    def fake_download_files_from_huggingface(
        hf_source_repo, cache_dir, extra_patterns, **kwargs
    ):
        local_files_only = bool(kwargs.get("local_files_only"))
        seen_local_files_only.append(local_files_only)
        if local_files_only:
            # The offline probe: hand back the already-populated snapshot dir.
            return str(snapshot_dir)
        # Reaching the online branch means the probe wrongly failed and a real
        # network fetch was attempted despite every required file being present.
        raise AssertionError("network fetch attempted despite cached files")

    monkeypatch.setattr(
        ModelManagement,
        "download_files_from_huggingface",
        staticmethod(fake_download_files_from_huggingface),
    )

    result = Bm25.download_model(model, str(tmp_path), local_files_only=False)

    assert Path(result) == snapshot_dir
    # Only the offline probe ran; the online (network) branch was never reached.
    assert seen_local_files_only == [True]
