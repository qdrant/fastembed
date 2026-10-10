"""Exercise streamed downloads over real HTTP without model downloads or credentials."""

import gzip
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

import pytest
import requests
from loguru import logger

from fastembed.common.model_management import ModelManagement
from fastembed.common.model_description import BaseModelDescription, ModelSource
from fastembed.common import model_management


@pytest.fixture
def download_server() -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            route = self.path.split("?", 1)[0]
            status = {"/forbidden": 403, "/missing": 404}.get(route, 200)
            body = gzip.compress(b"\0" * 10240) if route == "/empty-archive" else b"model"
            self.send_response(status)
            if route == "/invalid-length":
                self.send_header("Content-Length", "invalid")
            elif route == "/zero-length":
                self.send_header("Content-Length", "0")
            elif route != "/no-length":
                self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if route != "/zero-length":
                self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def download_logs() -> Iterator[list[str]]:
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)))
    try:
        yield messages
    finally:
        logger.remove(sink)


@pytest.fixture
def responses(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[requests.Response]]:
    """Capture real responses to observe their transport lifecycle, closing after failures."""
    captured: list[requests.Response] = []
    original_get = requests.get

    def get(*args, **kwargs):
        response = original_get(*args, **kwargs)
        captured.append(response)
        return response

    monkeypatch.setattr(requests, "get", get)
    try:
        yield captured
    finally:
        for response in captured:
            response.close()


@pytest.mark.parametrize(
    ("route", "error"),
    [
        ("/forbidden", PermissionError),
        ("/missing", requests.HTTPError),
        ("/invalid-length", ValueError),
    ],
)
def test_failed_download_closes_response(
    download_server: str, responses: list[requests.Response], tmp_path: Path, route, error
) -> None:
    destination = tmp_path / "model.tar.gz"
    destination.write_bytes(b"existing")

    with pytest.raises(error):
        ModelManagement.download_file_from_gcs(
            download_server + route, str(destination), show_progress=False
        )

    assert destination.read_bytes() == b"existing"
    assert responses[0].raw.closed, "failed download left its HTTP stream open"


def test_destination_error_closes_response(
    download_server: str, responses: list[requests.Response], tmp_path: Path
) -> None:
    with pytest.raises(OSError):
        ModelManagement.download_file_from_gcs(
            download_server + "/model", str(tmp_path), show_progress=False
        )

    assert responses[0].raw.closed, "an unwritable destination left its HTTP stream open"


@pytest.mark.parametrize("route", ["/no-length", "/zero-length"])
def test_download_warning_does_not_disclose_signed_url(
    download_server: str,
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
    download_logs: list[str],
    route: str,
) -> None:
    destination = tmp_path / "model.tar.gz"
    # This is a synthetic test marker, never a credential.
    url = download_server + route + "?signature=synthetic-private-marker"

    assert ModelManagement.download_file_from_gcs(url, str(destination)) == str(destination)

    output = capfd.readouterr()
    diagnostics = output.out + output.err + "".join(download_logs)
    assert "synthetic-private-marker" not in diagnostics
    assert "Content-length" in diagnostics
    assert destination.read_bytes() == (b"model" if route == "/no-length" else b"")


def test_successful_download_keeps_bytes_and_closes_response(
    download_server: str, responses: list[requests.Response], tmp_path: Path
) -> None:
    destination = tmp_path / "model.tar.gz"

    assert ModelManagement.download_file_from_gcs(
        download_server + "/model", str(destination), show_progress=False
    ) == str(destination)

    assert destination.read_bytes() == b"model"
    assert responses[0].raw.closed


def test_model_download_failure_does_not_log_signed_url(
    download_server: str,
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
    download_logs: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = download_server + "/missing?signature=synthetic-private-marker"
    model = BaseModelDescription(
        model="test/model",
        sources=ModelSource(url=url),
        model_file="model.onnx",
        description="Local HTTP fixture",
        license="Apache-2.0",
        size_in_GB=0.0,
    )
    # This test exercises a loopback URL, independently of external model offline settings.
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(model_management.time, "sleep", lambda _: None)

    with pytest.raises(ValueError, match="Could not load model test/model"):
        ModelManagement.download_model(model, str(tmp_path), retries=1)

    output = capfd.readouterr()
    diagnostics = output.out + output.err + "".join(download_logs)
    assert "synthetic-private-marker" not in diagnostics
    assert "Could not download model" in diagnostics


def test_incomplete_model_archive_error_omits_signed_url(
    download_server: str, tmp_path: Path
) -> None:
    url = download_server + "/empty-archive?signature=synthetic-private-marker"

    with pytest.raises(ValueError) as error:
        ModelManagement.retrieve_model_gcs("test/model", url, str(tmp_path))

    assert "synthetic-private-marker" not in str(error.value)
    assert "model directory" in str(error.value)
