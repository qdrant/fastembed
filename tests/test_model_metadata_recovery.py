"""Malformed cache metadata must not prevent normal snapshot resolution and repair."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from huggingface_hub.hf_api import RepoFile

from fastembed.common import model_management
from fastembed.common.model_description import BaseModelDescription, ModelSource
from fastembed.common.model_management import CorruptedCacheError, ModelManagement

REVISION = "0123456789abcdef0123456789abcdef01234567"
REPO = "qdrant/fake-onnx"
PAYLOAD = b"intact dummy model"


@pytest.fixture
def cached_model(tmp_path: Path) -> tuple[Path, Path, dict[str, dict[str, int | str]]]:
    repo_dir = tmp_path / "models--qdrant--fake-onnx"
    snapshot = repo_dir / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    model_file = snapshot / "model.onnx"
    model_file.write_bytes(PAYLOAD)
    metadata = {
        str(model_file.relative_to(repo_dir)): {"size": len(PAYLOAD), "blob_id": "modelblob"}
    }
    metadata_file = repo_dir / ModelManagement.METADATA_FILE
    metadata_file.write_text(json.dumps(metadata))
    return snapshot, metadata_file, metadata


@pytest.mark.parametrize("metadata_bytes", [b"", b"{", b"\xff"])
@pytest.mark.parametrize("local_files_only", [False, True])
def test_download_model_reuses_cache_with_malformed_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cached_model: tuple[Path, Path, dict[str, dict[str, int | str]]],
    metadata_bytes: bytes,
    local_files_only: bool,
) -> None:
    snapshot, metadata_file, _ = cached_model
    metadata_file.write_bytes(metadata_bytes)
    download = Mock(return_value=str(snapshot))
    info = Mock(side_effect=AssertionError("The cached model needs no hub request"))
    tree = Mock(side_effect=AssertionError("The cached model needs no hub request"))
    monkeypatch.setattr(model_management, "snapshot_download", download)
    monkeypatch.setattr(model_management, "model_info", info)
    monkeypatch.setattr(model_management, "list_repo_tree", tree)
    sleep = Mock()
    monkeypatch.setattr(model_management.time, "sleep", sleep)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    model = BaseModelDescription(
        model="test/fake",
        sources=ModelSource(hf=REPO),
        model_file="model.onnx",
        description="test",
        license="test",
        size_in_GB=0.0,
    )

    assert (
        ModelManagement.download_model(model, str(tmp_path), local_files_only=local_files_only)
        == snapshot
    )
    download.assert_called_once()
    assert download.call_args.kwargs["local_files_only"] is True
    assert not download.call_args.kwargs.get("force_download", False)
    info.assert_not_called()
    tree.assert_not_called()
    sleep.assert_not_called()
    assert (snapshot / "model.onnx").read_bytes() == PAYLOAD
    # Offline reuse does not manufacture authoritative metadata.
    assert metadata_file.read_bytes() == metadata_bytes


@pytest.mark.parametrize("metadata_bytes", [b"", b"{", b"\xff"])
@pytest.mark.parametrize("bad_size", [False, True])
def test_online_download_rebuilds_malformed_metadata_and_checks_sizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cached_model: tuple[Path, Path, dict[str, dict[str, int | str]]],
    metadata_bytes: bytes,
    bad_size: bool,
) -> None:
    snapshot, metadata_file, metadata = cached_model
    metadata_file.write_bytes(metadata_bytes)
    old_file = snapshot.parent / "older-revision" / "model.onnx"
    old_file.parent.mkdir()
    old_file.write_bytes(b"old")
    hub_file = RepoFile(path="model.onnx", size=len(PAYLOAD) + int(bad_size), oid="modelblob")
    download = Mock(return_value=str(snapshot))
    info = Mock(return_value=SimpleNamespace(sha=REVISION))
    tree = Mock(return_value=[hub_file])
    monkeypatch.setattr(model_management, "snapshot_download", download)
    monkeypatch.setattr(model_management, "model_info", info)
    monkeypatch.setattr(model_management, "list_repo_tree", tree)

    def load() -> str:
        return ModelManagement.download_files_from_huggingface(REPO, str(tmp_path), ["model.onnx"])

    if bad_size:
        with pytest.raises(ValueError, match="corrupted during downloading"):
            load()
        assert metadata_file.read_bytes() == metadata_bytes
    else:
        assert load() == str(snapshot)
        assert json.loads(metadata_file.read_text()) == metadata
    info.assert_called_once_with(REPO, timeout=model_management.constants.HF_HUB_ETAG_TIMEOUT)
    tree.assert_called_once_with(REPO, revision=REVISION, repo_type="model", recursive=True)
    download.assert_called_once()
    assert download.call_args.kwargs["local_files_only"] is False
    assert not download.call_args.kwargs.get("force_download", False)
    assert (snapshot / "model.onnx").read_bytes() == PAYLOAD
    assert old_file.read_bytes() == b"old"


@pytest.mark.parametrize("metadata_bytes", [b"{", b"\xff"])
def test_malformed_canonical_metadata_does_not_block_legacy_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cached_model: tuple[Path, Path, dict[str, dict[str, int | str]]],
    metadata_bytes: bytes,
) -> None:
    snapshot, _, _ = cached_model
    canonical_dir = tmp_path / "models--Qdrant--fake-onnx"
    if canonical_dir.exists():
        pytest.skip("Distinct repository casing requires a case-sensitive filesystem")
    canonical_dir.mkdir()
    (canonical_dir / ModelManagement.METADATA_FILE).write_bytes(metadata_bytes)
    download = Mock(side_effect=[FileNotFoundError("No canonical snapshot"), str(snapshot)])
    info = Mock(side_effect=AssertionError("Offline loading must not reach the hub"))
    tree = Mock(side_effect=AssertionError("Offline loading must not reach the hub"))
    monkeypatch.setattr(model_management, "snapshot_download", download)
    monkeypatch.setattr(model_management, "model_info", info)
    monkeypatch.setattr(model_management, "list_repo_tree", tree)

    assert ModelManagement.download_files_from_huggingface(
        "Qdrant/fake-onnx", str(tmp_path), ["model.onnx"], local_files_only=True
    ) == str(snapshot)
    assert [call.kwargs["repo_id"] for call in download.call_args_list] == [
        "Qdrant/fake-onnx",
        REPO,
    ]
    assert all(call.kwargs["local_files_only"] for call in download.call_args_list)
    info.assert_not_called()
    tree.assert_not_called()


@pytest.mark.parametrize("error", [PermissionError("denied"), ValueError("unrelated error")])
def test_metadata_reader_does_not_swallow_unrelated_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cached_model: tuple[Path, Path, dict[str, dict[str, int | str]]],
    error: Exception,
) -> None:
    def fail_read(*args: object, **kwargs: object) -> str:
        raise error

    monkeypatch.setattr(Path, "read_text", fail_read)
    with pytest.raises(type(error)) as exc:
        ModelManagement.download_files_from_huggingface(
            REPO, str(tmp_path), ["model.onnx"], local_files_only=True
        )
    assert exc.value is error


@pytest.mark.parametrize("metadata_text", ["[]", "null", '"invalid schema"'])
def test_valid_json_with_wrong_root_type_is_not_silently_accepted(
    tmp_path: Path,
    cached_model: tuple[Path, Path, dict[str, dict[str, int | str]]],
    metadata_text: str,
) -> None:
    _, metadata_file, _ = cached_model
    metadata_file.write_text(metadata_text)
    with pytest.raises(AttributeError):
        ModelManagement.download_files_from_huggingface(
            REPO, str(tmp_path), ["model.onnx"], local_files_only=True
        )


def test_valid_metadata_still_rejects_corrupt_model(
    tmp_path: Path,
    cached_model: tuple[Path, Path, dict[str, dict[str, int | str]]],
) -> None:
    snapshot, _, _ = cached_model
    (snapshot / "model.onnx").write_bytes(b"short")
    with pytest.raises(CorruptedCacheError):
        ModelManagement.download_files_from_huggingface(
            REPO, str(tmp_path), ["model.onnx"], local_files_only=True
        )


@pytest.mark.parametrize("local_files_only", [False, True])
def test_hub_failure_with_malformed_metadata_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cached_model: tuple[Path, Path, dict[str, dict[str, int | str]]],
    local_files_only: bool,
) -> None:
    _, metadata_file, _ = cached_model
    metadata_file.write_text("{")
    error = OSError("Hub unavailable")
    monkeypatch.setattr(model_management, "snapshot_download", Mock(side_effect=error))
    monkeypatch.setattr(
        model_management, "model_info", Mock(return_value=SimpleNamespace(sha=REVISION))
    )
    monkeypatch.setattr(model_management, "list_repo_tree", Mock(return_value=[]))
    with pytest.raises(OSError) as exc:
        ModelManagement.download_files_from_huggingface(
            REPO, str(tmp_path), ["model.onnx"], local_files_only=local_files_only
        )
    assert exc.value is error
