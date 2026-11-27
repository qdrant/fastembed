"""Download and cache-verification tests for ModelManagement.

The offline probe must distinguish a corrupt *model* file (which would make ONNX Runtime
fail with a cryptic protobuf error) from a benign size drift on an auxiliary file. A corrupt
model file must raise so ``download_model`` falls through to a forced re-download; an auxiliary
mismatch must be tolerated so an offline caller is not bricked by best-effort-loadable drift.
Model files may live in a subdirectory (e.g. ``onnx/model.onnx``), so matching is done on the
repo-relative path, not the bare filename.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, call

import pytest
from huggingface_hub.hf_api import RepoFile

from fastembed.common import model_management
from fastembed.common.model_description import BaseModelDescription, ModelSource
from fastembed.common.model_management import ModelManagement

REVISION = "0123456789abcdef0123456789abcdef01234567"


def _seed_cache(cache: Path, files: dict[str, bytes], metadata: dict[str, Any]) -> Path:
    snapshot = cache / "models--qdrant--fake-onnx"
    for name, blob in files.items():
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / ModelManagement.METADATA_FILE).write_text(json.dumps(metadata))
    return snapshot


def test_offline_probe_raises_when_model_file_is_corrupt(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    _seed_cache(
        cache,
        files={"model.onnx": b"x" * 10},
        metadata={"model.onnx": {"size": 999_999, "blob_id": "deadbeef"}},
    )

    with pytest.raises(ValueError, match="corrupt"):
        ModelManagement.download_files_from_huggingface(
            "qdrant/fake-onnx",
            cache_dir=str(cache),
            extra_patterns=["model.onnx"],
            local_files_only=True,
        )


def test_offline_probe_raises_when_subdir_model_file_is_corrupt(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    key = f"snapshots/{REVISION}/onnx/model.onnx"
    _seed_cache(
        cache,
        files={key: b"x" * 10},
        metadata={key: {"size": 999_999, "blob_id": "deadbeef"}},
    )

    with pytest.raises(ValueError, match="corrupt"):
        ModelManagement.download_files_from_huggingface(
            "qdrant/fake-onnx",
            cache_dir=str(cache),
            extra_patterns=["onnx/model.onnx"],
            local_files_only=True,
        )


def test_offline_probe_tolerates_auxiliary_file_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    snapshot = _seed_cache(
        cache,
        files={"model.onnx": b"x" * 10, "config.json": b"y" * 20},
        metadata={
            "model.onnx": {"size": 10, "blob_id": "modelblob"},
            "config.json": {"size": 999, "blob_id": "configblob"},
        },
    )

    # The model file matches metadata; only config.json drifted. The probe must NOT raise:
    # it falls through to snapshot_download (mocked here to return the cached path).
    monkeypatch.setattr(model_management, "snapshot_download", lambda **kwargs: str(snapshot))

    result = ModelManagement.download_files_from_huggingface(
        "qdrant/fake-onnx",
        cache_dir=str(cache),
        extra_patterns=["model.onnx"],
        local_files_only=True,
    )
    assert result == str(snapshot)


def test_online_download_ignores_older_cached_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_dir = tmp_path / "models--qdrant--fake-onnx"
    new_tokenizer = b"new tokenizer, longer than the old one"

    # an older revision is still in the cache
    old_file = repo_dir / "snapshots" / "old" / "tokenizer.json"
    old_file.parent.mkdir(parents=True)
    old_file.write_bytes(b"old tokenizer")

    # the hub has a newer revision, which the download puts next to the old one
    def fake_snapshot_download(**kwargs: Any) -> str:
        new_file = repo_dir / "snapshots" / "new" / "tokenizer.json"
        new_file.parent.mkdir(parents=True)
        new_file.write_bytes(new_tokenizer)
        return str(new_file.parent)

    hub_file = RepoFile(path="tokenizer.json", size=len(new_tokenizer), oid="new")
    monkeypatch.setattr(model_management, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(model_management, "list_repo_tree", lambda *a, **kw: [hub_file])
    monkeypatch.setattr(
        model_management, "model_info", lambda *a, **kw: SimpleNamespace(sha="new")
    )

    result = ModelManagement.download_files_from_huggingface(
        "qdrant/fake-onnx", cache_dir=str(tmp_path), extra_patterns=[]
    )

    assert result == str(repo_dir / "snapshots" / "new")
    # the size check ran on the new file, and only on it (keys use the OS path separator)
    metadata = json.loads((repo_dir / ModelManagement.METADATA_FILE).read_text())
    assert list(metadata) == [str(Path("snapshots/new/tokenizer.json"))]


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
