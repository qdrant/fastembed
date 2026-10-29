"""Download backoff belongs between attempts, never after the final failure."""

from pathlib import Path
from unittest.mock import Mock, call

import pytest

from fastembed.common import model_management
from fastembed.common.model_description import BaseModelDescription, ModelSource
from fastembed.common.model_management import ModelManagement


@pytest.mark.parametrize("retries,delays", [(1, []), (3, [3.0, 9.0])])
@pytest.mark.parametrize("with_url_fallback", [False, True])
def test_download_failure_only_sleeps_between_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retries: int,
    delays: list[float],
    with_url_fallback: bool,
) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    source = ModelSource(
        hf="test/missing-model",
        url="https://example.invalid/model.tar.gz" if with_url_fallback else None,
    )
    model = BaseModelDescription(
        model="test/missing-model",
        sources=source,
        model_file="model.onnx",
        description="Test fixture",
        license="Apache-2.0",
        size_in_GB=0.0,
    )
    download = Mock(side_effect=OSError("download unavailable"))
    fallback = Mock(side_effect=OSError("fallback unavailable"))
    sleep = Mock()
    monkeypatch.setattr(ModelManagement, "download_files_from_huggingface", download)
    monkeypatch.setattr(ModelManagement, "retrieve_model_gcs", fallback)
    monkeypatch.setattr(model_management.time, "sleep", sleep)

    with pytest.raises(ValueError, match="Could not load model test/missing-model"):
        ModelManagement.download_model(model, str(tmp_path), retries=retries)

    # One offline cache probe, followed by exactly the requested download attempts.
    assert download.call_count == retries + 1
    assert fallback.call_count == (retries if with_url_fallback else 0)
    assert sleep.call_args_list == [call(delay) for delay in delays]
