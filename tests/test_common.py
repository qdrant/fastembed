import io
import os
import tarfile

import pytest

from fastembed import (
    TextEmbedding,
    SparseTextEmbedding,
    ImageEmbedding,
    LateInteractionMultimodalEmbedding,
    LateInteractionTextEmbedding,
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


def test_decompress_to_cache_blocks_path_traversal(tmp_path):
    targz_path = tmp_path / "evil.tar.gz"
    escape_target = tmp_path / "escaped.txt"
    with tarfile.open(targz_path, "w:gz") as tar:
        payload = b"PWNED"
        info = tarfile.TarInfo(name=f"../{escape_target.name}")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    with pytest.raises(ValueError):
        ModelManagement.decompress_to_cache(str(targz_path), str(cache_dir))
    assert not escape_target.exists()


def test_decompress_to_cache_extracts_safe_archive(tmp_path):
    targz_path = tmp_path / "safe.tar.gz"
    with tarfile.open(targz_path, "w:gz") as tar:
        payload = b"model bytes"
        info = tarfile.TarInfo(name="model/config.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    ModelManagement.decompress_to_cache(str(targz_path), str(cache_dir))
    assert (cache_dir / "model" / "config.json").read_bytes() == b"model bytes"
    assert os.path.isdir(cache_dir)
