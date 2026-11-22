import os

# onnxruntime queues telemetry for Microsoft from its first import on; opt out before it.
# A user's own ORT_DISABLE_TELEMETRY (e.g. 0) is kept.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
