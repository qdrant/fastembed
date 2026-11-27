"""Exercise real Hub snapshot/ref cache handling with only network boundaries replaced."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import huggingface_hub.file_download as file_download
from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile
import pytest

from fastembed.common import model_management
from fastembed.common.model_management import ModelManagement

COMMIT = "a" * 40
CONTENTS = b"small local model fixture"


@pytest.fixture
def hub_transport(monkeypatch: pytest.MonkeyPatch) -> Callable[[], None]:
    """Supply metadata and HTTP bytes; keep snapshot_download and hf_hub_download real."""
    tree = [RepoFile(path="model.onnx", size=len(CONTENTS), oid="b" * 40)]
    monkeypatch.setattr(
        model_management, "model_info", lambda *a, **kw: SimpleNamespace(sha=COMMIT)
    )
    monkeypatch.setattr(model_management, "list_repo_tree", lambda *a, **kw: tree)
    monkeypatch.setattr(
        HfApi,
        "repo_info",
        lambda *a, **kw: SimpleNamespace(
            sha=COMMIT, siblings=[SimpleNamespace(rfilename="model.onnx")]
        ),
    )
    monkeypatch.setattr(HfApi, "list_repo_tree", lambda *a, **kw: tree)

    def metadata(url: str, *args: Any, **kwargs: Any) -> Any:
        """Return a simulated HTTP HEAD result while enforcing the resolved-commit URL."""
        assert f"/resolve/{COMMIT}/" in url
        fields: dict[str, Any] = dict(
            commit_hash=COMMIT,
            etag="b" * 40,
            location="https://example.invalid/model.onnx",
            size=len(CONTENTS),
        )
        if "xet_file_data" in file_download.HfFileMetadata.__annotations__:
            fields["xet_file_data"] = None
        return file_download.HfFileMetadata(**fields)

    def http_get(url: str, temp_file: Any, **kwargs: Any) -> None:
        """Provide response bytes; real Hub code creates blobs, snapshots and ref files."""
        temp_file.write(CONTENTS)

    monkeypatch.setattr(file_download, "get_hf_file_metadata", metadata)
    monkeypatch.setattr(file_download, "http_get", http_get)

    def disable_network() -> None:
        """Turn every replaced network boundary into an error for the offline phase."""

        def forbidden(*args: Any, **kwargs: Any) -> Any:
            """Fail if local-only cache resolution tries any metadata or HTTP request."""
            raise AssertionError("Offline lookup touched a network boundary")

        monkeypatch.setattr(model_management, "model_info", forbidden)
        monkeypatch.setattr(model_management, "list_repo_tree", forbidden)
        monkeypatch.setattr(HfApi, "repo_info", forbidden)
        monkeypatch.setattr(HfApi, "list_repo_tree", forbidden)
        monkeypatch.setattr(file_download, "get_hf_file_metadata", forbidden)
        monkeypatch.setattr(file_download, "http_get", forbidden)

    return disable_network


@pytest.mark.parametrize("revision", [None, "release-v1", "feature/topic", COMMIT])
def test_pinned_online_download_remains_loadable_by_original_ref_offline(
    tmp_path: Path, hub_transport: Callable[[], None], revision: str | None
) -> None:
    """A real snapshot lookup must resolve the original ref without a second network request."""
    kwargs = {} if revision is None else {"revision": revision}
    online = ModelManagement.download_files_from_huggingface(
        "qdrant/fake-onnx", cache_dir=str(tmp_path), extra_patterns=["model.onnx"], **kwargs
    )
    assert Path(online).name == COMMIT
    assert (Path(online) / "model.onnx").read_bytes() == CONTENTS
    hub_transport()

    offline = ModelManagement.download_files_from_huggingface(
        "qdrant/fake-onnx",
        cache_dir=str(tmp_path),
        extra_patterns=["model.onnx"],
        local_files_only=True,
        **kwargs,
    )

    assert offline == online
    refs = tmp_path / "models--qdrant--fake-onnx" / "refs"
    if revision == COMMIT:
        assert not refs.exists()
    else:
        assert (refs / (revision or "main")).read_text() == COMMIT


@pytest.mark.parametrize("correct_ref_exists", [True, False])
def test_ref_write_failure_does_not_fail_cached_online_download(
    tmp_path: Path,
    hub_transport: Callable[[], None],
    monkeypatch: pytest.MonkeyPatch,
    correct_ref_exists: bool,
) -> None:
    """A correct ref needs no write; failure to update one must not discard a downloaded model."""
    online = ModelManagement.download_files_from_huggingface(
        "qdrant/fake-onnx", cache_dir=str(tmp_path), extra_patterns=["model.onnx"]
    )
    refs = tmp_path / "models--qdrant--fake-onnx" / "refs"
    if not correct_ref_exists:
        (refs / "main").unlink()
    replacements: list[str] = []

    def readonly(source: Any, destination: Any) -> None:
        """Simulate a denied atomic rename after the snapshot already exists."""
        replacements.append(str(destination))
        raise PermissionError("read-only cache ref")

    monkeypatch.setattr(model_management.os, "replace", readonly)

    repeated = ModelManagement.download_files_from_huggingface(
        "qdrant/fake-onnx", cache_dir=str(tmp_path), extra_patterns=["model.onnx"]
    )

    assert repeated == online
    assert replacements == ([] if correct_ref_exists else [str(refs / "main")])
    assert list(refs.iterdir()) == ([refs / "main"] if correct_ref_exists else [])


@pytest.mark.parametrize("revision", ["../outside", "a/../../outside", "/absolute", "C:\\outside"])
def test_ref_mapping_does_not_leave_cache(tmp_path: Path, revision: str) -> None:
    """Cache bookkeeping must not turn a supplied revision into an arbitrary write path."""
    snapshot = tmp_path / "snapshots" / COMMIT
    snapshot.mkdir(parents=True)
    model_management._cache_requested_revision(tmp_path, str(snapshot), revision, COMMIT)
    assert sorted(tmp_path.iterdir()) == [tmp_path / "snapshots"]


def test_ref_mapping_does_not_follow_external_symlink(tmp_path: Path) -> None:
    """An existing symlink outside refs must not redirect a cache write."""
    snapshot = tmp_path / "snapshots" / COMMIT
    snapshot.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    refs = tmp_path / "refs"
    try:
        refs.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("This platform does not permit directory symlinks")
    model_management._cache_requested_revision(tmp_path, str(snapshot), "main", COMMIT)
    assert not list(outside.iterdir())


def test_local_dir_download_does_not_create_snapshot_ref(
    tmp_path: Path, hub_transport: Callable[[], None]
) -> None:
    """A local_dir download is not evidence of a snapshot being available offline in cache_dir."""
    cache = tmp_path / "cache"
    local = tmp_path / "local"
    result = ModelManagement.download_files_from_huggingface(
        "qdrant/fake-onnx",
        cache_dir=str(cache),
        extra_patterns=["model.onnx"],
        local_dir=str(local),
    )
    assert Path(result) == local
    assert not (cache / "models--qdrant--fake-onnx" / "refs").exists()
