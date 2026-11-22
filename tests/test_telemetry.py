import os
import subprocess
import sys

import pytest

# Each case runs in a fresh interpreter: fastembed sets the opt-out at import time.
CHECK_ENV = "import os, fastembed; print(os.environ.get('ORT_DISABLE_TELEMETRY'))"
# onnxruntime imported before fastembed: fastembed must call disable_telemetry_events().
CHECK_API = (
    "import onnxruntime as ort\n"
    "calls = []\n"
    "ort.disable_telemetry_events = lambda: calls.append(1)\n"
    "import fastembed\n"
    "print(len(calls))"
)


def run_python(code: str, ort_disable_telemetry: str | None) -> str:
    env = {k: v for k, v in os.environ.items() if k != "ORT_DISABLE_TELEMETRY"}
    if ort_disable_telemetry is not None:
        env["ORT_DISABLE_TELEMETRY"] = ort_disable_telemetry
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def test_telemetry_disabled_by_default() -> None:
    assert run_python(CHECK_ENV, None) == "1"


@pytest.mark.parametrize("value", ["0", "false", "1"])
def test_user_setting_is_kept(value: str) -> None:
    assert run_python(CHECK_ENV, value) == value


@pytest.mark.parametrize(("value", "calls"), [(None, "1"), ("1", "1"), ("0", "0"), ("false", "0")])
def test_disable_telemetry_events_when_onnxruntime_imported_first(
    value: str | None, calls: str
) -> None:
    assert run_python(CHECK_API, value) == calls
