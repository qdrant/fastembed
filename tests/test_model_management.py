"""Cache-verification tests for ModelManagement.

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


@pytest.mark.parametrize("revision", ["release-v1", "a" * 40])
def test_online_download_verifies_the_requested_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, revision: str
) -> None:
    """A valid pinned snapshot must not be compared with the current main's file sizes."""
    old_revision, main_revision = "a" * 40, "b" * 40
    contents = {old_revision: b"old model", main_revision: b"new model with a different size"}
    repo_dir = tmp_path / "models--qdrant--fake-onnx"
    lookups: list[str | None] = []

    def fake_model_info(repo_id: str, **kwargs: Any) -> SimpleNamespace:
        """Resolve the requested ref to its old commit; an unqualified lookup sees main."""
        requested = kwargs.get("revision")
        lookups.append(requested)
        return SimpleNamespace(sha=old_revision if requested == revision else main_revision)

    def fake_repo_tree(repo_id: str, **kwargs: Any) -> list[RepoFile]:
        """Return metadata for the given commit from fixtures with different file sizes."""
        commit = kwargs["revision"]
        return [RepoFile(path="model.onnx", size=len(contents[commit]), oid=commit)]

    def fake_snapshot_download(**kwargs: Any) -> str:
        """Write the selected snapshot so production verification checks real local bytes."""
        requested = kwargs.get("revision")
        commit = old_revision if requested in (revision, old_revision) else main_revision
        model_file = repo_dir / "snapshots" / commit / "model.onnx"
        model_file.parent.mkdir(parents=True)
        model_file.write_bytes(contents[commit])
        return str(model_file.parent)

    monkeypatch.setattr(model_management, "model_info", fake_model_info)
    monkeypatch.setattr(model_management, "list_repo_tree", fake_repo_tree)
    monkeypatch.setattr(model_management, "snapshot_download", fake_snapshot_download)

    result = ModelManagement.download_files_from_huggingface(
        "qdrant/fake-onnx",
        cache_dir=str(tmp_path),
        extra_patterns=["model.onnx"],
        revision=revision,
    )

    assert result == str(repo_dir / "snapshots" / old_revision)
    assert lookups == [revision]
    metadata = json.loads((repo_dir / ModelManagement.METADATA_FILE).read_text())
    assert metadata[str(Path("snapshots") / old_revision / "model.onnx")]["size"] == len(
        contents[old_revision]
    )


@pytest.mark.parametrize("revision", [None, "stable"])
def test_online_download_pins_the_revision_resolved_for_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, revision: str | None
) -> None:
    """A moving ref must not select different files after their metadata was fetched."""
    resolved_revision, moved_revision = "a" * 40, "b" * 40
    contents = {resolved_revision: b"old model", moved_revision: b"new model, new size"}
    repo_dir = tmp_path / "models--qdrant--fake-onnx"
    downloads: list[str | None] = []

    monkeypatch.setattr(
        model_management, "model_info", lambda *a, **kw: SimpleNamespace(sha=resolved_revision)
    )
    monkeypatch.setattr(
        model_management,
        "list_repo_tree",
        lambda *a, **kw: [
            RepoFile(
                path="model.onnx", size=len(contents[resolved_revision]), oid=resolved_revision
            )
        ],
    )

    def fake_snapshot_download(**kwargs: Any) -> str:
        """Simulate a ref moving after metadata lookup; only its resolved SHA keeps old bytes."""
        requested = kwargs.get("revision")
        downloads.append(requested)
        # The branch has moved since model_info: only a commit SHA still selects the old file.
        commit = resolved_revision if requested == resolved_revision else moved_revision
        model_file = repo_dir / "snapshots" / commit / "model.onnx"
        model_file.parent.mkdir(parents=True)
        model_file.write_bytes(contents[commit])
        return str(model_file.parent)

    monkeypatch.setattr(model_management, "snapshot_download", fake_snapshot_download)

    result = ModelManagement.download_files_from_huggingface(
        "qdrant/fake-onnx",
        cache_dir=str(tmp_path),
        extra_patterns=["model.onnx"],
        revision=revision,
    )

    assert result == str(repo_dir / "snapshots" / resolved_revision)
    assert downloads == [resolved_revision]
