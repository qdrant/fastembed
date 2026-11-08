import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from loguru import logger

from fastembed.common.utils import _is_writable_dir, define_cache_dir, get_default_cache_dir


def test_define_cache_dir_explicit_path(tmp_path: Path) -> None:
    custom_dir = tmp_path / "custom_cache"
    result = define_cache_dir(str(custom_dir))
    assert result == custom_dir
    assert result.is_dir()


def test_define_cache_dir_env_var_override(tmp_path: Path) -> None:
    env_dir = tmp_path / "env_cache"
    with patch.dict(os.environ, {"FASTEMBED_CACHE_PATH": str(env_dir)}):
        result = define_cache_dir()
        assert result == env_dir
        assert result.is_dir()


def test_get_default_cache_dir_windows(tmp_path: Path) -> None:
    local_app_data = tmp_path / "LocalAppData"
    with (
        patch.dict(os.environ, {"LOCALAPPDATA": str(local_app_data)}, clear=False),
        patch("sys.platform", "win32"),
    ):
        os.environ.pop("FASTEMBED_CACHE_PATH", None)
        result = get_default_cache_dir()
        expected = local_app_data / "fastembed_cache"
        assert result == expected
        assert result.is_dir()


def test_get_default_cache_dir_darwin(tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    with (
        patch("pathlib.Path.home", return_value=fake_home),
        patch("sys.platform", "darwin"),
        patch.dict(os.environ, {}, clear=False),
    ):
        os.environ.pop("FASTEMBED_CACHE_PATH", None)
        result = get_default_cache_dir()
        expected = fake_home / "Library" / "Caches" / "fastembed_cache"
        assert result == expected
        assert result.is_dir()


def test_get_default_cache_dir_linux_xdg(tmp_path: Path) -> None:
    fake_xdg = tmp_path / "xdg_cache"
    with (
        patch.dict(os.environ, {"XDG_CACHE_HOME": str(fake_xdg)}, clear=False),
        patch("sys.platform", "linux"),
    ):
        os.environ.pop("FASTEMBED_CACHE_PATH", None)
        result = get_default_cache_dir()
        expected = fake_xdg / "fastembed_cache"
        assert result == expected
        assert result.is_dir()


def test_get_default_cache_dir_linux_fallback_home(tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    with (
        patch("pathlib.Path.home", return_value=fake_home),
        patch("sys.platform", "linux"),
        patch.dict(os.environ, {}, clear=False),
    ):
        os.environ.pop("FASTEMBED_CACHE_PATH", None)
        os.environ.pop("XDG_CACHE_HOME", None)
        result = get_default_cache_dir()
        expected = fake_home / ".cache" / "fastembed_cache"
        assert result == expected
        assert result.is_dir()


def test_get_default_cache_dir_permission_error_fallback() -> None:
    call_count = 0

    def fake_mkdir(*args: object, **kwargs: object) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise PermissionError("Permission denied on persistent user cache")

    with (
        patch.dict(os.environ, {}, clear=False),
        patch("sys.platform", "linux"),
        patch("pathlib.Path.mkdir", side_effect=fake_mkdir),
    ):
        os.environ.pop("FASTEMBED_CACHE_PATH", None)
        fallback_expected = Path(tempfile.gettempdir()) / "fastembed_cache"
        result = get_default_cache_dir()
        assert result == fallback_expected
        assert call_count == 2


def test_get_default_cache_dir_existing_non_writable_directory_fallback(tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    non_writable_cache = fake_home / ".cache" / "fastembed_cache"
    non_writable_cache.mkdir(parents=True, exist_ok=True)

    with (
        patch("pathlib.Path.home", return_value=fake_home),
        patch("sys.platform", "linux"),
        patch.dict(os.environ, {}, clear=False),
        patch(
            "fastembed.common.utils._is_writable_dir",
            side_effect=lambda p: str(non_writable_cache) not in str(p),
        ),
    ):
        os.environ.pop("FASTEMBED_CACHE_PATH", None)
        os.environ.pop("XDG_CACHE_HOME", None)
        fallback_expected = Path(tempfile.gettempdir()) / "fastembed_cache"
        result = get_default_cache_dir()
        assert result == fallback_expected


def test_is_writable_dir_success_and_cleanup(tmp_path: Path) -> None:
    test_dir = tmp_path / "writable_check"
    test_dir.mkdir(parents=True, exist_ok=True)
    assert _is_writable_dir(test_dir) is True
    # Verify no probe files are left behind
    leftover = list(test_dir.glob(".fastembed_write_test_*"))
    assert leftover == []


def test_is_writable_dir_permission_denied(tmp_path: Path) -> None:
    test_dir = tmp_path / "denied_check"
    test_dir.mkdir(parents=True, exist_ok=True)
    with patch("pathlib.Path.touch", side_effect=PermissionError("Mocked denial")):
        assert _is_writable_dir(test_dir) is False
    leftover = list(test_dir.glob(".fastembed_write_test_*"))
    assert leftover == []


def test_get_default_cache_dir_fallback_mkdir_fails() -> None:
    def fake_mkdir(*args: object, **kwargs: object) -> None:
        raise PermissionError("All mkdir calls denied")

    with (
        patch.dict(os.environ, {}, clear=False),
        patch("sys.platform", "linux"),
        patch("pathlib.Path.mkdir", side_effect=fake_mkdir),
    ):
        os.environ.pop("FASTEMBED_CACHE_PATH", None)
        with pytest.raises(PermissionError, match="neither persistent.*nor fallback"):
            get_default_cache_dir()


def test_define_cache_dir_fallback_failure_raises() -> None:
    def fake_mkdir(*args: object, **kwargs: object) -> None:
        raise PermissionError("All mkdir calls denied")

    with (
        patch.dict(os.environ, {}, clear=False),
        patch("sys.platform", "linux"),
        patch("pathlib.Path.mkdir", side_effect=fake_mkdir),
    ):
        os.environ.pop("FASTEMBED_CACHE_PATH", None)
        with pytest.raises(PermissionError, match="neither persistent.*nor fallback"):
            define_cache_dir(None)


def test_get_default_cache_dir_logs_warning_on_fallback() -> None:
    call_count = 0

    def fake_mkdir(*args: object, **kwargs: object) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise PermissionError("Persistent cache unwritable")

    with (
        patch.dict(os.environ, {}, clear=False),
        patch("sys.platform", "linux"),
        patch("pathlib.Path.mkdir", side_effect=fake_mkdir),
        patch.object(logger, "warning") as mock_warn,
    ):
        os.environ.pop("FASTEMBED_CACHE_PATH", None)
        fallback_expected = Path(tempfile.gettempdir()) / "fastembed_cache"
        result = get_default_cache_dir()
        assert result == fallback_expected
        assert mock_warn.called
        assert "Falling back to temporary cache directory" in mock_warn.call_args[0][0]
