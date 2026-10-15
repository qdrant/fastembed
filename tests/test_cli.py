from pathlib import Path
from unittest.mock import Mock

import pytest

from fastembed.cli import main
from fastembed.common.types import Device


@pytest.fixture(autouse=True)
def server_dependencies() -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")


def test_serve_loads_one_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = Mock()
    factory = Mock(return_value=model)
    create_app = Mock()
    run = Mock()
    monkeypatch.setattr("fastembed.TextEmbedding", factory)
    monkeypatch.setattr("fastembed.server.create_app", create_app)
    monkeypatch.setattr("uvicorn.run", run)
    main(
        [
            "serve",
            "--model-name",
            "test-model",
            "--model-path",
            str(tmp_path),
            "--device",
            "cuda",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--threads",
            "2",
            "--batch-size",
            "16",
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )
    factory.assert_called_once_with(
        model_name="test-model",
        specific_model_path=str(tmp_path),
        cuda=Device.CUDA,
        cache_dir=str(tmp_path / "cache"),
        threads=2,
    )
    create_app.assert_called_once_with(model, batch_size=16)
    run.assert_called_once_with(create_app.return_value, host="0.0.0.0", port=9000)


def test_serve_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    factory, create_app, run = Mock(), Mock(), Mock()
    monkeypatch.setattr("fastembed.TextEmbedding", factory)
    monkeypatch.setattr("fastembed.server.create_app", create_app)
    monkeypatch.setattr("uvicorn.run", run)
    main(["serve"])
    factory.assert_called_once_with(cuda=Device.CPU, cache_dir=None, threads=None)
    run.assert_called_once_with(create_app.return_value, host="127.0.0.1", port=8080)


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["serve", "--port", "0"],
        ["serve", "--port", "65536"],
        ["serve", "--batch-size", "-1"],
        ["serve", "--threads", "0"],
        ["serve", "--device", "invalid"],
        ["serve", "--model-path", "missing-directory"],
    ],
)
def test_invalid_arguments(args: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2


def test_model_load_error(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr("fastembed.TextEmbedding", Mock(side_effect=ValueError("invalid model")))
    run = Mock()
    monkeypatch.setattr("uvicorn.run", run)
    with pytest.raises(SystemExit) as exc:
        main(["serve", "--model-name", "not-supported"])
    assert exc.value.code == 1
    assert "Unable to load the embedding model" in capsys.readouterr().err
    run.assert_not_called()
