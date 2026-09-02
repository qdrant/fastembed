import re
from unittest.mock import patch

import numpy as np
import pytest

from fastembed import (
    TextEmbedding,
    SparseTextEmbedding,
    ImageEmbedding,
    LateInteractionMultimodalEmbedding,
    LateInteractionTextEmbedding,
)
from fastembed.common.model_description import BaseModelDescription, ModelSource
from fastembed.common.model_management import ModelManagement
from fastembed.common.utils import last_token_pooling


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


def _make_local_model_dir(root, model_file="onnx/model.onnx", additional_files=()):
    """Create a directory that looks like a downloaded model snapshot."""
    for rel_path in ("config.json", "tokenizer.json", model_file, *additional_files):
        file_path = root / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("{}")
    return root


def _local_model_description(hf_source, model_file="onnx/model.onnx", additional_files=None):
    return BaseModelDescription(
        model="test-org/test-model",
        sources=ModelSource(hf=hf_source),
        model_file=model_file,
        description="",
        license="",
        size_in_GB=0.1,
        additional_files=additional_files or [],
    )


def test_local_directory_hf_source_is_used_as_is(tmp_path):
    """An `hf` source pointing to a local directory must be used without touching the hub."""
    model_dir = _make_local_model_dir(tmp_path / "my-model")
    model = _local_model_description(str(model_dir))

    with patch.object(ModelManagement, "download_files_from_huggingface") as mock_download:
        resolved_path = ModelManagement.download_model(model, cache_dir=str(tmp_path / "cache"))

    assert resolved_path == model_dir
    mock_download.assert_not_called()


def test_local_directory_hf_source_expands_user(tmp_path, monkeypatch):
    """`~` in a local directory source must be expanded."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # windows
    model_dir = _make_local_model_dir(tmp_path / "my-model")
    model = _local_model_description("~/my-model")

    with patch.object(ModelManagement, "download_files_from_huggingface") as mock_download:
        resolved_path = ModelManagement.download_model(model, cache_dir=str(tmp_path / "cache"))

    assert resolved_path == model_dir
    mock_download.assert_not_called()


def test_local_directory_hf_source_with_additional_files(tmp_path):
    """`additional_files` must be resolved relative to the local directory as well."""
    model_dir = _make_local_model_dir(
        tmp_path / "my-model", additional_files=("stopwords/en.txt",)
    )
    model = _local_model_description(str(model_dir), additional_files=["stopwords/en.txt"])

    resolved_path = ModelManagement.download_model(model, cache_dir=str(tmp_path / "cache"))

    assert resolved_path == model_dir


def test_local_directory_hf_source_missing_files(tmp_path):
    """An incomplete local directory must fail loudly instead of falling back to the hub."""
    model_dir = _make_local_model_dir(tmp_path / "my-model")
    model = _local_model_description(
        str(model_dir), additional_files=["stopwords/en.txt", "vocab.txt"]
    )

    with patch.object(ModelManagement, "download_files_from_huggingface") as mock_download:
        with pytest.raises(ValueError, match="stopwords/en.txt, vocab.txt"):
            ModelManagement.download_model(model, cache_dir=str(tmp_path / "cache"))

    mock_download.assert_not_called()


def test_local_directory_hf_source_rejects_path_traversal(tmp_path):
    """A required file escaping the local directory (via `..` or an absolute path) must not resolve."""
    model_dir = _make_local_model_dir(tmp_path / "my-model")
    outside_file = tmp_path / "outside.onnx"
    outside_file.write_text("{}")

    traversal_model = _local_model_description(str(model_dir), model_file="../outside.onnx")
    absolute_model = _local_model_description(str(model_dir), model_file=str(outside_file))

    with patch.object(ModelManagement, "download_files_from_huggingface") as mock_download:
        with pytest.raises(ValueError, match=re.escape("../outside.onnx")):
            ModelManagement.download_model(traversal_model, cache_dir=str(tmp_path / "cache"))
        with pytest.raises(ValueError, match=re.escape(str(outside_file))):
            ModelManagement.download_model(absolute_model, cache_dir=str(tmp_path / "cache"))

    mock_download.assert_not_called()


def test_repo_id_hf_source_is_not_treated_as_local_directory(tmp_path):
    """A repo id, which does not resolve to a directory, must still go through the hub."""
    model = _local_model_description("test-org/test-model")
    snapshot_dir = _make_local_model_dir(tmp_path / "snapshot")

    with patch.object(
        ModelManagement, "download_files_from_huggingface", return_value=str(snapshot_dir)
    ) as mock_download:
        resolved_path = ModelManagement.download_model(model, cache_dir=str(tmp_path / "cache"))

    assert resolved_path == snapshot_dir
    mock_download.assert_called_with(
        "test-org/test-model",
        cache_dir=str(tmp_path / "cache"),
        extra_patterns=["onnx/model.onnx"],
        local_files_only=True,
    )
