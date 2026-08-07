import os
from unittest.mock import MagicMock, patch

from fastembed import (
    ImageEmbedding,
    LateInteractionMultimodalEmbedding,
    LateInteractionTextEmbedding,
    SparseTextEmbedding,
    TextEmbedding,
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


def _run_download_with_mocks(tmp_path, extra_env):
    """Run download_files_from_huggingface with all network calls mocked out.

    Uses autospec=True so mocks are checked against the real huggingface_hub
    signatures - a kwarg that the real API doesn't accept (e.g. passing
    `endpoint` to a bound HfApi method instead of its constructor) raises a
    TypeError here, the same as it would against the real library.
    """
    mock_hf_api_instance = MagicMock()
    mock_hf_api_instance.model_info.return_value = MagicMock(sha="abc123")
    mock_hf_api_instance.list_repo_tree.return_value = []

    with (
        patch.dict(os.environ, extra_env),
        patch(
            "fastembed.common.model_management.HfApi",
            autospec=True,
            return_value=mock_hf_api_instance,
        ) as mock_hf_api_cls,
        patch(
            "fastembed.common.model_management.snapshot_download",
            autospec=True,
            return_value=str(tmp_path),
        ) as mock_sd,
        # skip post-download metadata verification so the function completes cleanly
        patch.object(ModelManagement, "METADATA_FILE", "__nonexistent__"),
    ):
        ModelManagement.download_files_from_huggingface(
            hf_source_repo="test-org/test-model",
            cache_dir=str(tmp_path),
            extra_patterns=["*.onnx"],
        )
        return mock_hf_api_cls, mock_hf_api_instance, mock_sd


def test_hf_endpoint_forwarded_to_hub_calls(tmp_path):
    """HF_ENDPOINT env var must be forwarded to HfApi and snapshot_download."""
    custom_endpoint = "https://hf-mirror.example.com"
    mock_hf_api_cls, mock_hf_api_instance, mock_sd = _run_download_with_mocks(
        tmp_path, {"HF_ENDPOINT": custom_endpoint}
    )

    _, api_kwargs = mock_hf_api_cls.call_args
    assert api_kwargs.get("endpoint") == custom_endpoint, (
        f"HfApi should be constructed with endpoint={custom_endpoint!r}, got {api_kwargs}"
    )
    mock_hf_api_instance.model_info.assert_called_once()
    mock_hf_api_instance.list_repo_tree.assert_called_once()

    _, sd_kwargs = mock_sd.call_args
    assert sd_kwargs.get("endpoint") == custom_endpoint, (
        f"snapshot_download should receive endpoint={custom_endpoint!r}, got {sd_kwargs}"
    )


def test_no_hf_endpoint_no_extra_kwarg(tmp_path):
    """When HF_ENDPOINT is not set, endpoint must be None for HfApi and snapshot_download."""
    env_without_hf_endpoint = {k: v for k, v in os.environ.items() if k != "HF_ENDPOINT"}
    with patch.dict(os.environ, env_without_hf_endpoint, clear=True):
        mock_hf_api_cls, _, mock_sd = _run_download_with_mocks(tmp_path, {})

    _, api_kwargs = mock_hf_api_cls.call_args
    assert api_kwargs.get("endpoint") is None, (
        f"HfApi endpoint should be None when HF_ENDPOINT is unset, got {api_kwargs}"
    )

    _, sd_kwargs = mock_sd.call_args
    assert sd_kwargs.get("endpoint") is None, (
        f"snapshot_download endpoint should be None when HF_ENDPOINT is unset, got {sd_kwargs}"
    )
