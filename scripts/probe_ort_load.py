"""Load a ColModernVBERT model.onnx with one graph optimization level and run a tiny text query.

TEMPORARY, delete before merging. Usage: python -X faulthandler -u probe_ort_load.py <repo> <all|basic>
"""

import sys

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

repo, level = sys.argv[1], sys.argv[2]
levels = {
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
}
path = hf_hub_download(repo, "model.onnx")
print(f"onnxruntime {ort.__version__}, {repo}, optimization {level}: loading", flush=True)
options = ort.SessionOptions()
options.graph_optimization_level = levels[level]
session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
print("loaded", flush=True)
ids = np.array([[50281, 25521, 1533, 50282]], dtype=np.int64)
output = session.run(
    None,
    {
        "input_ids": ids,
        "attention_mask": np.ones_like(ids),
        "pixel_values": np.zeros((1, 1, 3, 512, 512), np.float32),
    },
)[0]
print(f"query ok, output {output.shape}", flush=True)
