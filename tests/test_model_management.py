"""Offline cache-verification tests for ModelManagement.

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

import pytest
from huggingface_hub.hf_api import RepoFile

from fastembed.common import model_management
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


def test_online_verification_only_collects_downloaded_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cache = Path("cache")
    old_key = f"snapshots/{'a' * 40}/onnx/model.onnx"
    current_key = f"snapshots/{REVISION}/onnx/model.onnx"
    repo_dir = _seed_cache(
        cache,
        files={old_key: b"old", current_key: b"current"},
        metadata={old_key: {"size": 3, "blob_id": "old-blob"}},
    )
    current_snapshot = (repo_dir / "snapshots" / REVISION).resolve()
    monkeypatch.setattr(
        model_management, "model_info", lambda *args, **kwargs: SimpleNamespace(sha=REVISION)
    )
    monkeypatch.setattr(
        model_management,
        "list_repo_tree",
        lambda *args, **kwargs: [RepoFile(path="onnx/model.onnx", size=7, oid="current-blob")],
    )
    monkeypatch.setattr(
        model_management, "snapshot_download", lambda **kwargs: str(current_snapshot)
    )

    result = ModelManagement.download_files_from_huggingface(
        "qdrant/fake-onnx", str(cache), extra_patterns=["onnx/model.onnx"]
    )
    assert result == str(current_snapshot)
    assert json.loads((repo_dir / ModelManagement.METADATA_FILE).read_text()) == {
        current_key: {"size": 7, "blob_id": "current-blob"}
    }
    assert (repo_dir / old_key).read_bytes() == b"old"
