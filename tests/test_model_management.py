import io
import tarfile

import pytest

from fastembed.common.model_management import ModelManagement


def _add_file(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def test_decompress_to_cache_extracts_safe_members(tmp_path):
    archive_path = tmp_path / "model.tar.gz"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    with tarfile.open(archive_path, "w:gz") as tar:
        _add_file(tar, "model/config.json", b"{}")

    assert ModelManagement.decompress_to_cache(str(archive_path), str(cache_dir)) == str(cache_dir)
    assert (cache_dir / "model" / "config.json").read_bytes() == b"{}"


def test_decompress_to_cache_rejects_member_path_traversal(tmp_path):
    archive_path = tmp_path / "model.tar.gz"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    outside_path = tmp_path / "outside.txt"

    with tarfile.open(archive_path, "w:gz") as tar:
        _add_file(tar, "../outside.txt", b"owned")

    with pytest.raises(ValueError, match="Unsafe tar member path"):
        ModelManagement.decompress_to_cache(str(archive_path), str(cache_dir))

    assert not outside_path.exists()


def test_decompress_to_cache_rejects_symlink_path_traversal(tmp_path):
    archive_path = tmp_path / "model.tar.gz"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    with tarfile.open(archive_path, "w:gz") as tar:
        link = tarfile.TarInfo(name="model/escape")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../outside.txt"
        tar.addfile(link)

    with pytest.raises(ValueError, match="Unsafe tar link target"):
        ModelManagement.decompress_to_cache(str(archive_path), str(cache_dir))
