"""Offline cache-verification tests for ModelManagement.

The offline probe must distinguish a corrupt *model* file (which would make ONNX Runtime
fail with a cryptic protobuf error) from a benign size drift on an auxiliary file. A corrupt
model file must raise so ``download_model`` falls through to a forced re-download; an auxiliary
mismatch must be tolerated so an offline caller is not bricked by best-effort-loadable drift.
Model files may live in a subdirectory (e.g. ``onnx/model.onnx``), so matching is done on the
repo-relative path, not the bare filename.
"""

import json
import shutil
import tarfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

import onnx
import pytest
from onnx import TensorProto as T
from onnx import helper as h
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from fastembed import TextEmbedding
from fastembed.common import model_management
from fastembed.common.model_description import ModelSource, PoolingType
from fastembed.common.model_management import ModelManagement
from fastembed.text.custom_text_embedding import CustomTextEmbedding


@pytest.fixture(autouse=True)
def restore_custom_models_fixture():
    supported_models = CustomTextEmbedding.SUPPORTED_MODELS
    postprocessing_mapping = CustomTextEmbedding.POSTPROCESSING_MAPPING
    CustomTextEmbedding.SUPPORTED_MODELS = []
    CustomTextEmbedding.POSTPROCESSING_MAPPING = {}
    yield
    CustomTextEmbedding.SUPPORTED_MODELS = supported_models
    CustomTextEmbedding.POSTPROCESSING_MAPPING = postprocessing_mapping


def _build_url_model_archive(
    root: Path, name: str, factor: float, archive_dir: str = "shared-model"
) -> None:
    """Builds a tiny two-dimensional ONNX model and packs it as ``<name>.tar.gz``.

    The archive's top-level directory is ``archive_dir`` (default
    "shared-model", mirroring a URL-backed custom model whose basename
    collides with another model's). The ``factor`` distinguishes the two
    models: embedding "hello world" (token ids 2 and 3, summing to 5)
    yields [5.0, 5.0 * factor].
    """
    folder = root / name / archive_dir
    folder.mkdir(parents=True)
    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "[PAD]": 1, "hello": 2, "world": 3}, unk_token="<redacted>")
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.enable_padding(pad_id=1, pad_token="<redacted>")
    tokenizer.enable_truncation(max_length=8)
    tokenizer.save(str(folder / "tokenizer.json"))
    (folder / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 8, "pad_token": "[PAD]"})
    )
    (folder / "special_tokens_map.json").write_text(
        json.dumps({"unk_token": "[UNK]", "pad_token": "[PAD]"})
    )
    graph = h.make_graph(
        [
            h.make_node("Cast", ["input_ids"], ["tokens"], to=T.FLOAT),
            h.make_node("ReduceSum", ["tokens"], ["sum"], axes=[1], keepdims=1),
            h.make_node("Mul", ["sum", "factor"], ["scaled"]),
            h.make_node("Concat", ["sum", "scaled"], ["embedding"], axis=1),
        ],
        name,
        [h.make_tensor_value_info("input_ids", T.INT64, ["batch", "sequence"])],
        [h.make_tensor_value_info("embedding", T.FLOAT, ["batch", 2])],
        [h.make_tensor("factor", T.FLOAT, [1], [factor])],
    )
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 11)], ir_version=9)
    onnx.checker.check_model(model)
    onnx.save(model, str(folder / "model.onnx"))
    with tarfile.open(root / f"{name}.tar.gz", "w:gz") as out:
        out.add(folder, arcname=archive_dir)


def _register_url_models(base_url: str) -> None:
    for owner in ("a", "b"):
        TextEmbedding.add_custom_model(
            model=f"owner-{owner}/shared-model",
            pooling=PoolingType.DISABLED,
            normalization=False,
            sources=ModelSource(url=f"{base_url}/{owner}.tar.gz"),
            dim=2,
            model_file="model.onnx",
        )


def _embed_first(model_name: str, cache_dir: Path, **kwargs: Any) -> list[float]:
    model = TextEmbedding(model_name, cache_dir=str(cache_dir), threads=1, **kwargs)
    return next(model.embed(["hello world"])).tolist()


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


def test_url_models_with_same_basename_use_separate_cache_dirs(tmp_path: Path) -> None:
    root = tmp_path / "server"
    root.mkdir()
    _build_url_model_archive(root, "a", 1.0)
    _build_url_model_archive(root, "b", -1.0)
    cache = tmp_path / "shared"

    with ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=root)
    ) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            _register_url_models(f"http://127.0.0.1:{server.server_port}")
            assert _embed_first("owner-a/shared-model", cache) == [5.0, 5.0]
            # Previously returned [5.0, 5.0]: the second model reused the first
            # model's files from the shared basename cache directory.
            assert _embed_first("owner-b/shared-model", cache) == [5.0, -5.0]
        finally:
            server.shutdown()
            thread.join()

    assert (cache / "namespaced--owner%2Da--shared%2Dmodel" / "model.onnx").exists()
    assert (cache / "namespaced--owner%2Db--shared%2Dmodel" / "model.onnx").exists()
    assert not (cache / "shared-model").exists()


def test_url_models_with_double_dash_names_use_separate_cache_dirs(tmp_path: Path) -> None:
    root = tmp_path / "server"
    root.mkdir()
    _build_url_model_archive(root, "p", 1.0, archive_dir="b--c")
    _build_url_model_archive(root, "q", -1.0, archive_dir="c")
    cache = tmp_path / "shared"

    with ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=root)
    ) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            for model, archive in (("x/b--c", "p"), ("x--b/c", "q")):
                TextEmbedding.add_custom_model(
                    model=model,
                    pooling=PoolingType.DISABLED,
                    normalization=False,
                    sources=ModelSource(
                        url=f"http://127.0.0.1:{server.server_port}/{archive}.tar.gz"
                    ),
                    dim=2,
                    model_file="model.onnx",
                )
            # Under a plain "/" -> "--" encoding both names map to "x--b--c",
            # so the second load would silently reuse the first model's files.
            assert _embed_first("x/b--c", cache) == [5.0, 5.0]
            assert _embed_first("x--b/c", cache) == [5.0, -5.0]
        finally:
            server.shutdown()
            thread.join()

    assert (cache / "namespaced--x--b%2D%2Dc" / "model.onnx").exists()
    assert (cache / "namespaced--x%2D%2Db--c" / "model.onnx").exists()


def test_url_models_with_dash_next_to_slash_use_separate_cache_dirs(tmp_path: Path) -> None:
    root = tmp_path / "server"
    root.mkdir()
    _build_url_model_archive(root, "p", 1.0)
    _build_url_model_archive(root, "q", -1.0)
    cache = tmp_path / "shared"

    with ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=root)
    ) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            for model, archive in (("a-/b/shared-model", "p"), ("a/-b/shared-model", "q")):
                TextEmbedding.add_custom_model(
                    model=model,
                    pooling=PoolingType.DISABLED,
                    normalization=False,
                    sources=ModelSource(
                        url=f"http://127.0.0.1:{server.server_port}/{archive}.tar.gz"
                    ),
                    dim=2,
                    model_file="model.onnx",
                )
            # A "-" next to a "/" used to collide too: both names mapped to
            # "a---b--shared-model", so the second load silently reused the
            # first model's files.
            assert _embed_first("a-/b/shared-model", cache) == [5.0, 5.0]
            assert _embed_first("a/-b/shared-model", cache) == [5.0, -5.0]
        finally:
            server.shutdown()
            thread.join()

    assert (cache / "namespaced--a%2D--b--shared%2Dmodel" / "model.onnx").exists()
    assert (cache / "namespaced--a--%2Db--shared%2Dmodel" / "model.onnx").exists()


def test_url_model_ignores_legacy_dir_sharing_its_cache_key_when_online(tmp_path: Path) -> None:
    root = tmp_path / "server"
    root.mkdir()
    _build_url_model_archive(root, "p", 1.0, archive_dir="model")
    _build_url_model_archive(root, "q", -1.0, archive_dir="owner--model")
    cache = tmp_path / "shared"
    # A legacy basename-only copy of "other/owner--model": its directory is
    # "owner--model", exactly the key the old encoding gave "owner/model",
    # so an online load used to accept it as a cache hit and load the other
    # model's weights.
    shutil.copytree(root / "q" / "owner--model", cache / "owner--model")

    with ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=root)
    ) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            TextEmbedding.add_custom_model(
                model="owner/model",
                pooling=PoolingType.DISABLED,
                normalization=False,
                sources=ModelSource(url=f"http://127.0.0.1:{server.server_port}/p.tar.gz"),
                dim=2,
                model_file="model.onnx",
            )
            assert _embed_first("owner/model", cache) == [5.0, 5.0]
        finally:
            server.shutdown()
            thread.join()

    assert (cache / "namespaced--owner--model" / "model.onnx").exists()
    # The legacy directory is left untouched for its own model.
    assert (cache / "owner--model" / "model.onnx").exists()


def test_url_model_ignores_legacy_basename_cache_dir_when_online(tmp_path: Path) -> None:
    root = tmp_path / "server"
    root.mkdir()
    _build_url_model_archive(root, "a", 1.0)
    _build_url_model_archive(root, "b", -1.0)
    cache = tmp_path / "shared"
    # A legacy basename-only copy of a's files, as an older fastembed version cached it.
    shutil.copytree(root / "a" / "shared-model", cache / "shared-model")

    with ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=root)
    ) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            _register_url_models(f"http://127.0.0.1:{server.server_port}")
            assert _embed_first("owner-b/shared-model", cache) == [5.0, -5.0]
        finally:
            server.shutdown()
            thread.join()

    assert (cache / "namespaced--owner%2Db--shared%2Dmodel" / "model.onnx").exists()


def test_url_model_rejects_legacy_dir_sharing_its_namespaced_cache_key_when_online(
    tmp_path: Path,
) -> None:
    root = tmp_path / "server"
    root.mkdir()
    _build_url_model_archive(root, "p", 1.0, archive_dir="model")
    _build_url_model_archive(root, "q", -1.0, archive_dir="namespaced--owner--model")
    cache = tmp_path / "shared"
    # A legacy basename-only copy of "other/namespaced--owner--model": its
    # directory is "namespaced--owner--model", exactly the namespaced key
    # of "owner/model", so the prefix alone cannot tell them apart and an
    # online load used to accept it as a cache hit and load the other
    # model's weights.
    shutil.copytree(root / "q" / "namespaced--owner--model", cache / "namespaced--owner--model")

    with ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=root)
    ) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            TextEmbedding.add_custom_model(
                model="owner/model",
                pooling=PoolingType.DISABLED,
                normalization=False,
                sources=ModelSource(url=f"http://127.0.0.1:{server.server_port}/p.tar.gz"),
                dim=2,
                model_file="model.onnx",
            )
            # Previously returned [5.0, -5.0]: the squatting legacy directory
            # was returned as a cache hit for "owner/model".
            assert _embed_first("owner/model", cache) == [5.0, 5.0]
        finally:
            server.shutdown()
            thread.join()

    entry = cache / "namespaced--owner--model"
    assert (entry / "model.onnx").exists()
    assert (entry / ModelManagement.CACHE_PROVENANCE_FILENAME).read_text(
        encoding="utf-8"
    ) == "owner/model"
    # The squatter is moved aside, not deleted: its files survive for
    # manual recovery.
    backups = list(cache.glob(".namespaced--owner--model.squatter-*"))
    assert len(backups) == 1
    assert (backups[0] / "model.onnx").exists()


def test_url_model_refuses_legacy_squatter_when_offline(tmp_path: Path) -> None:
    root = tmp_path / "server"
    root.mkdir()
    _build_url_model_archive(root, "q", -1.0, archive_dir="namespaced--owner--model")
    cache = tmp_path / "shared"
    # Only the squatting legacy directory exists and there is no server to
    # fetch from: the load must fail loudly rather than silently return
    # the other model's weights.
    shutil.copytree(root / "q" / "namespaced--owner--model", cache / "namespaced--owner--model")

    TextEmbedding.add_custom_model(
        model="owner/model",
        pooling=PoolingType.DISABLED,
        normalization=False,
        sources=ModelSource(url="http://127.0.0.1:1/p.tar.gz"),
        dim=2,
        model_file="model.onnx",
    )

    with pytest.raises(ValueError, match="Could not load model owner/model"):
        _embed_first("owner/model", cache, local_files_only=True)


def test_url_model_reuses_namespaced_entry_with_provenance_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "server"
    root.mkdir()
    _build_url_model_archive(root, "p", 1.0, archive_dir="model")
    cache = tmp_path / "shared"

    with ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=root)
    ) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            TextEmbedding.add_custom_model(
                model="owner/model",
                pooling=PoolingType.DISABLED,
                normalization=False,
                sources=ModelSource(url=f"http://127.0.0.1:{server.server_port}/p.tar.gz"),
                dim=2,
                model_file="model.onnx",
            )
            assert _embed_first("owner/model", cache) == [5.0, 5.0]
        finally:
            server.shutdown()
            thread.join()

    entry = cache / "namespaced--owner--model"
    assert (entry / ModelManagement.CACHE_PROVENANCE_FILENAME).read_text(
        encoding="utf-8"
    ) == "owner/model"

    # The marker-bearing entry must be accepted as a cache hit: a second
    # load must not attempt a download at all.
    def _must_not_download(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("must not re-download a marker-bearing cache entry")

    monkeypatch.setattr(ModelManagement, "download_file_from_gcs", _must_not_download)
    assert _embed_first("owner/model", cache) == [5.0, 5.0]


def test_url_model_uses_legacy_basename_cache_dir_when_offline(tmp_path: Path) -> None:
    root = tmp_path / "server"
    root.mkdir()
    _build_url_model_archive(root, "a", 1.0)
    cache = tmp_path / "shared"
    # Only a legacy basename-only copy exists and there is no server to fetch
    # from, so the legacy copy is the only usable one.
    shutil.copytree(root / "a" / "shared-model", cache / "shared-model")

    TextEmbedding.add_custom_model(
        model="owner-a/shared-model",
        pooling=PoolingType.DISABLED,
        normalization=False,
        sources=ModelSource(url="http://127.0.0.1:1/a.tar.gz"),
        dim=2,
        model_file="model.onnx",
    )

    assert _embed_first("owner-a/shared-model", cache, local_files_only=True) == [5.0, 5.0]
