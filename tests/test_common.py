import os
from unittest.mock import patch, MagicMock

from fastembed import (
    TextEmbedding,
    SparseTextEmbedding,
    ImageEmbedding,
    LateInteractionMultimodalEmbedding,
    LateInteractionTextEmbedding,
)
from fastembed.common.model_management import ModelManagement


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


def test_hf_endpoint_forwarded_to_hub_calls(tmp_path):
    """HF_ENDPOINT env var must be forwarded to model_info, list_repo_tree, and snapshot_download."""
    custom_endpoint = "https://hf-mirror.example.com"

    mock_model_info = MagicMock()
    mock_model_info.sha = "abc123"

    mock_repo_file = MagicMock()
    mock_repo_file.path = "model.onnx"
    mock_repo_file.__class__ = __import__(
        "huggingface_hub.hf_api", fromlist=["RepoFile"]
    ).RepoFile

    with (
        patch.dict(os.environ, {"HF_ENDPOINT": custom_endpoint}),
        patch("fastembed.common.model_management.model_info", return_value=mock_model_info) as mock_mi,
        patch("fastembed.common.model_management.list_repo_tree", return_value=[]) as mock_lrt,
        patch(
            "fastembed.common.model_management.snapshot_download",
            return_value=str(tmp_path),
        ) as mock_sd,
    ):
        try:
            ModelManagement.download_files_from_huggingface(
                hf_source_repo="test-org/test-model",
                cache_dir=str(tmp_path),
                extra_patterns=["*.onnx"],
            )
        except Exception:
            pass  # we only care that the calls received the right kwargs

        mock_mi.assert_called_once()
        _, mi_kwargs = mock_mi.call_args
        assert mi_kwargs.get("endpoint") == custom_endpoint, (
            f"model_info should receive endpoint={custom_endpoint!r}, got {mi_kwargs}"
        )

        mock_lrt.assert_called_once()
        _, lrt_kwargs = mock_lrt.call_args
        assert lrt_kwargs.get("endpoint") == custom_endpoint, (
            f"list_repo_tree should receive endpoint={custom_endpoint!r}, got {lrt_kwargs}"
        )

        mock_sd.assert_called_once()
        _, sd_kwargs = mock_sd.call_args
        assert sd_kwargs.get("endpoint") == custom_endpoint, (
            f"snapshot_download should receive endpoint={custom_endpoint!r}, got {sd_kwargs}"
        )


def test_no_hf_endpoint_no_extra_kwarg(tmp_path):
    """When HF_ENDPOINT is not set, endpoint kwarg must NOT be passed to hub calls."""
    mock_model_info = MagicMock()
    mock_model_info.sha = "abc123"

    env_without_hf_endpoint = {k: v for k, v in os.environ.items() if k != "HF_ENDPOINT"}

    with (
        patch.dict(os.environ, env_without_hf_endpoint, clear=True),
        patch("fastembed.common.model_management.model_info", return_value=mock_model_info) as mock_mi,
        patch("fastembed.common.model_management.list_repo_tree", return_value=[]) as mock_lrt,
        patch(
            "fastembed.common.model_management.snapshot_download",
            return_value=str(tmp_path),
        ) as mock_sd,
    ):
        try:
            ModelManagement.download_files_from_huggingface(
                hf_source_repo="test-org/test-model",
                cache_dir=str(tmp_path),
                extra_patterns=["*.onnx"],
            )
        except Exception:
            pass

        _, mi_kwargs = mock_mi.call_args
        assert "endpoint" not in mi_kwargs, "endpoint kwarg should not be present when HF_ENDPOINT is unset"

        _, lrt_kwargs = mock_lrt.call_args
        assert "endpoint" not in lrt_kwargs, "endpoint kwarg should not be present when HF_ENDPOINT is unset"

        _, sd_kwargs = mock_sd.call_args
        assert "endpoint" not in sd_kwargs, "endpoint kwarg should not be present when HF_ENDPOINT is unset"
