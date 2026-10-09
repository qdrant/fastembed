"""Offline integrity checks must apply to the snapshot Hugging Face actually selects."""

import json
from pathlib import Path
from typing import Any

import pytest
from huggingface_hub import snapshot_download

from fastembed.common.model_description import BaseModelDescription, ModelSource
from fastembed.common.model_management import CorruptedCacheError, ModelManagement

OLD_REVISION = "0" * 40
CURRENT_REVISION = "1" * 40
REPO_ID = "qdrant/fake-onnx"
MODEL_FILE = "onnx/model.onnx"


def _seed_revisions(cache: Path, *, corrupt_current: bool = False) -> Path:
    repo = cache / "models--qdrant--fake-onnx"
    metadata = {}
    for revision, contents, recorded_size in [
        (OLD_REVISION, b"old truncated model", 100),
        (CURRENT_REVISION, b"current model", 100 if corrupt_current else 13),
    ]:
        file = repo / "snapshots" / revision / MODEL_FILE
        file.parent.mkdir(parents=True)
        file.write_bytes(contents)
        metadata[str(file.relative_to(repo))] = {
            "size": recorded_size,
            "blob_id": f"blob-{revision}",
        }
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(CURRENT_REVISION)
    (repo / "refs" / "release").write_text(CURRENT_REVISION)
    (repo / ModelManagement.METADATA_FILE).write_text(json.dumps(metadata))
    return repo / "snapshots" / CURRENT_REVISION


@pytest.mark.parametrize("revision", [None, "release", CURRENT_REVISION])
@pytest.mark.parametrize("manifest_has_current", [False, True])
def test_offline_probe_ignores_corruption_in_another_revision(
    tmp_path: Path, revision: str | None, manifest_has_current: bool
) -> None:
    current = _seed_revisions(tmp_path)
    metadata_file = current.parent.parent / ModelManagement.METADATA_FILE
    if not manifest_has_current:
        metadata = json.loads(metadata_file.read_text())
        del metadata[str(Path("snapshots") / CURRENT_REVISION / MODEL_FILE)]
        metadata_file.write_text(json.dumps(metadata))

    # A real hub cache lookup selects a complete current snapshot without any network calls.
    expected = snapshot_download(
        repo_id=REPO_ID, cache_dir=tmp_path, local_files_only=True, revision=revision
    )
    assert expected == str(current)
    assert (current / MODEL_FILE).read_bytes() == b"current model"

    assert (
        ModelManagement.download_files_from_huggingface(
            REPO_ID,
            cache_dir=str(tmp_path),
            extra_patterns=[MODEL_FILE],
            local_files_only=True,
            revision=revision,
        )
        == expected
    )


@pytest.mark.parametrize("revision", [None, "release", CURRENT_REVISION])
def test_offline_probe_still_rejects_corruption_in_selected_revision(
    tmp_path: Path, revision: str | None
) -> None:
    _seed_revisions(tmp_path, corrupt_current=True)
    with pytest.raises(CorruptedCacheError):
        ModelManagement.download_files_from_huggingface(
            REPO_ID,
            cache_dir=str(tmp_path),
            extra_patterns=[MODEL_FILE],
            local_files_only=True,
            revision=revision,
        )


def test_offline_download_model_ignores_corruption_in_another_revision(tmp_path: Path) -> None:
    current = _seed_revisions(tmp_path)
    model = BaseModelDescription(
        model=REPO_ID,
        sources=ModelSource(hf=REPO_ID),
        model_file=MODEL_FILE,
        description="test model",
        license="",
        size_in_GB=0,
    )
    assert ModelManagement.download_model(model, str(tmp_path), local_files_only=True) == current


def test_offline_probe_keeps_corruption_error_when_revision_cannot_be_resolved(
    tmp_path: Path,
) -> None:
    current = _seed_revisions(tmp_path)
    (current.parent.parent / "refs" / "main").unlink()
    # Preserve the signal used by download_model to force-refetch corrupt cached blobs.
    with pytest.raises(CorruptedCacheError):
        ModelManagement.download_files_from_huggingface(
            REPO_ID,
            cache_dir=str(tmp_path),
            extra_patterns=[MODEL_FILE],
            local_files_only=True,
        )


def test_offline_probe_falls_back_from_corrupt_current_to_legacy_casing(tmp_path: Path) -> None:
    current = _seed_revisions(tmp_path, corrupt_current=True)
    legacy_repo = tmp_path / "models--Qdrant--Fake-onnx"
    if legacy_repo.exists():
        pytest.skip("requires a case-sensitive filesystem")
    legacy_current = _seed_revisions(tmp_path / "legacy-cache")
    legacy_current.parent.parent.rename(legacy_repo)
    expected = legacy_repo / "snapshots" / CURRENT_REVISION
    assert ModelManagement.download_files_from_huggingface(
        REPO_ID,
        cache_dir=str(tmp_path),
        extra_patterns=[MODEL_FILE],
        local_files_only=True,
    ) == str(expected)
    assert (current / MODEL_FILE).stat().st_size == 13


def test_download_model_forces_redownload_of_corrupt_selected_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    current = _seed_revisions(tmp_path, corrupt_current=True)
    model = BaseModelDescription(
        model=REPO_ID,
        sources=ModelSource(hf=REPO_ID),
        model_file=MODEL_FILE,
        description="test model",
        license="",
        size_in_GB=0,
    )
    real_download = ModelManagement.download_files_from_huggingface
    online_calls: list[dict[str, Any]] = []

    def download(repo_id: str, **kwargs: Any) -> str:
        if kwargs.get("local_files_only"):
            return real_download(repo_id, **kwargs)
        online_calls.append(kwargs)
        return str(current)

    monkeypatch.setattr(ModelManagement, "download_files_from_huggingface", download)
    assert ModelManagement.download_model(model, str(tmp_path)) == current
    assert len(online_calls) == 1
    assert online_calls[0]["force_download"] is True


def test_offline_probe_keeps_existing_checks_for_local_dir(tmp_path: Path) -> None:
    _seed_revisions(tmp_path)
    local_dir = tmp_path / "standalone"
    local_dir.mkdir()
    (local_dir / "model.onnx").write_bytes(b"standalone model")
    resolved = snapshot_download(
        repo_id=REPO_ID,
        cache_dir=tmp_path,
        local_dir=local_dir,
        local_files_only=True,
    )
    if Path(resolved) != local_dir:
        pytest.skip("this hub version does not resolve local_dir for offline loads")
    # The directory name is not a snapshot revision. Preserve existing validation
    # instead of interpreting it as a revision and discarding all cached metadata.
    with pytest.raises(CorruptedCacheError):
        ModelManagement.download_files_from_huggingface(
            REPO_ID,
            cache_dir=str(tmp_path),
            extra_patterns=[MODEL_FILE],
            local_files_only=True,
            local_dir=local_dir,
        )
