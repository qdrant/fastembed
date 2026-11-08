import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastembed.common.utils import define_cache_dir, get_default_cache_dir


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
