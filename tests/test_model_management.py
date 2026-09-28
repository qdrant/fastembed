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
from typing import Any

import pytest

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


"""huggingface-hub version-compatibility tests (issue #742).

These tests run fully offline and lock in support for huggingface-hub 2.x:
the import surface fastembed uses, the transport-error shim resolving to the
hub's own HTTP library, and the metadata-verified download flow with a hub
2.x-style RepoFile.
"""


def _make_repo_file(path: str, size: int, blob_id: str):
    """Build a RepoFile the way the installed huggingface_hub does.

    hub 2.x constructs RepoFile from API payload keys (``oid`` instead of
    ``blob_id``); older versions take ``blob_id=`` directly.
    """
    from fastembed.common import model_management

    try:
        return model_management.RepoFile(path=path, size=size, blob_id=blob_id)
    except (TypeError, KeyError):
        return model_management.RepoFile(path=path, size=size, oid=blob_id)


def test_hf_import_surface_resolves() -> None:
    """Every huggingface_hub name model_management imports must exist."""
    import huggingface_hub
    from huggingface_hub import constants
    from huggingface_hub.file_download import repo_folder_name
    from huggingface_hub.utils import (
        HFValidationError,
        RepositoryNotFoundError,
        disable_progress_bars,
        enable_progress_bars,
    )

    from fastembed.common import model_management

    assert huggingface_hub.__version__
    assert constants.REPO_ID_SEPARATOR == "--"
    assert isinstance(constants.HF_HUB_ETAG_TIMEOUT, (int, float))
    assert callable(model_management.snapshot_download)
    assert callable(model_management.model_info)
    assert callable(model_management.list_repo_tree)
    assert callable(repo_folder_name)
    for name in (
        HFValidationError,
        RepositoryNotFoundError,
        disable_progress_bars,
        enable_progress_bars,
    ):
        assert name is not None

    repo_file = _make_repo_file("onnx/model.onnx", 10, "deadbeef")
    assert (repo_file.path, repo_file.size, repo_file.blob_id) == (
        "onnx/model.onnx",
        10,
        "deadbeef",
    )


def test_transport_error_shim_catches_hub_http_errors() -> None:
    """_hf_transport_errors() must cover the hub's own HTTP library's errors."""
    import pytest

    from fastembed.common import model_management

    errors = model_management._hf_transport_errors()
    assert isinstance(errors, tuple)
    assert all(isinstance(err, type) and issubclass(err, Exception) for err in errors)
    assert OSError in model_management._HF_DOWNLOAD_ERRORS
    assert ValueError in model_management._HF_DOWNLOAD_ERRORS
    from huggingface_hub.utils import RepositoryNotFoundError

    assert RepositoryNotFoundError in model_management._HF_DOWNLOAD_ERRORS

    if errors:
        # hub >= 1.0 (httpx) and 2.x (httpx2) raise TransportError, not OSError,
        # on refused connections / DNS failures / timeouts.
        with pytest.raises(model_management._HF_DOWNLOAD_ERRORS):
            raise errors[0]("simulated transport failure")


def test_online_download_flow_mocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full download path works with hub 2.x-style RepoFile metadata, offline.

    model_info/list_repo_tree/snapshot_download are mocked so no network is
    needed; the test exercises the metadata collection and verification logic
    against a RepoFile built the way hub 2.x builds one.
    """
    from types import SimpleNamespace

    from fastembed.common import model_management

    cache = tmp_path / "cache"
    blobs = {"model.onnx": b"x" * 10, "config.json": b"y" * 20}
    snapshot = _seed_cache(cache, files=blobs, metadata={})
    # no pre-existing metadata: the online flow must collect it from the repo files
    (snapshot / ModelManagement.METADATA_FILE).unlink()

    monkeypatch.setattr(
        model_management,
        "model_info",
        lambda repo_id, **kwargs: SimpleNamespace(sha=REVISION),
    )
    monkeypatch.setattr(
        model_management,
        "list_repo_tree",
        lambda repo_id, **kwargs: [
            _make_repo_file("model.onnx", len(blobs["model.onnx"]), "modelblob"),
            _make_repo_file("config.json", len(blobs["config.json"]), "configblob"),
        ],
    )
    monkeypatch.setattr(model_management, "snapshot_download", lambda **kwargs: str(snapshot))

    result = ModelManagement.download_files_from_huggingface(
        "qdrant/fake-onnx",
        cache_dir=str(cache),
        extra_patterns=["model.onnx"],
        local_files_only=False,
    )
    assert result == str(snapshot)

    # the download wrote back the collected metadata, verified against the repo files
    metadata = json.loads((snapshot / ModelManagement.METADATA_FILE).read_text())
    assert metadata["model.onnx"] == {"size": 10, "blob_id": "modelblob"}
    assert metadata["config.json"] == {"size": 20, "blob_id": "configblob"}
