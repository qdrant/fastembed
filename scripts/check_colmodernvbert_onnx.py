"""Runtime check of a ColModernVBERT model.onnx with the installed fastembed and onnxruntime.

Downloads the model from a Hugging Face revision (a branch that fastembed users never resolve), then checks:
  1. fastembed's canonical image and query values (atol 2e-3), through the installed fastembed's own code path
  2. text queries give the same output with one blank tile per token (fastembed <= 0.8.x), 1 tile and 0 tiles
  3. an image embedded inside a zero-padded batch equals the same image embedded alone
Exits 1 on any failure.

Usage: python check_colmodernvbert_onnx.py --revision <branch> [--repo Qdrant/colmodernvbert]
       python check_colmodernvbert_onnx.py --model-dir <local dir>
"""

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from PIL import Image

import fastembed
from fastembed import LateInteractionMultimodalEmbedding

# tests/test_late_interaction_multimodal.py
CANONICAL_IMAGE = np.array(
    [
        [0.11614, -0.15793, -0.11194, 0.0688, 0.08001, 0.10575, -0.07871],
        [0.10094, -0.13301, -0.12069, 0.10932, 0.04645, 0.09884, 0.04048],
        [0.13106, -0.18613, -0.13469, 0.10566, 0.03659, 0.07712, -0.03916],
        [0.09754, -0.09596, -0.04839, 0.14991, 0.05692, 0.10569, -0.08349],
        [0.02576, -0.15651, -0.09977, 0.09707, 0.13412, 0.09994, -0.09931],
        [-0.06741, -0.1787, -0.19677, -0.07618, 0.13102, -0.02131, -0.02437],
        [-0.02776, -0.10187, -0.13793, 0.03835, 0.04766, 0.04701, -0.15635],
    ]
)
CANONICAL_QUERY = np.array(
    [
        [0.05, 0.06557, 0.04026, 0.14981, 0.1842, 0.0263, -0.18706],
        [-0.05664, -0.14028, 0.00649, -0.02849, 0.09034, -0.01494, 0.10693],
        [-0.10147, -0.00716, 0.09084, -0.08236, -0.01849, -0.00972, -0.00461],
        [-0.1233, -0.10814, -0.02337, -0.00329, 0.05984, 0.09934, 0.09846],
        [-0.07053, -0.13119, -0.06487, 0.01508, 0.07459, 0.07655, 0.14821],
        [0.00526, -0.13842, -0.05837, -0.02721, 0.13009, 0.05076, 0.17962],
        [0.00924, -0.14383, -0.03057, -0.03691, 0.11718, 0.037, 0.13344],
    ]
)
QUERIES = ["hello world", "flag embedding"]
TEST_IMAGE = Path(__file__).resolve().parents[1] / "tests" / "misc" / "image.jpeg"

parser = argparse.ArgumentParser()
parser.add_argument("--repo", default="Qdrant/colmodernvbert")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--revision", help="Hugging Face branch or commit to download")
group.add_argument("--model-dir", help="local directory instead of a download")
args = parser.parse_args()

model_dir = args.model_dir or snapshot_download(
    args.repo, revision=args.revision, allow_patterns=["model.onnx", "*.json"]
)
print(
    f"python {platform.python_version()} {platform.system()}-{platform.machine()}, onnxruntime {ort.__version__}, "
    f"fastembed {fastembed.__version__}, model {args.model_dir or f'{args.repo}@{args.revision}'}"
)
failures = []


def check(name, diff, tol, seconds):
    status = "ok" if diff <= tol else "FAIL"
    if status == "FAIL":
        failures.append(name)
    print(f"  {name}: max diff {diff:.1e} (tol {tol:g}) {status} [{seconds:.2f}s]")


model = LateInteractionMultimodalEmbedding("Qdrant/colmodernvbert", specific_model_path=model_dir)
image = Image.open(TEST_IMAGE).convert("RGB")
wide = image.crop(
    (0, 0, image.width, image.width // 4)
)  # 5 tiles instead of 13, so batches get padding

t = time.perf_counter()
single = list(model.embed_image([str(TEST_IMAGE)], batch_size=1))[0]
check(
    "canonical image",
    float(np.abs(single[:7, :7] - CANONICAL_IMAGE).max()),
    2e-3,
    time.perf_counter() - t,
)

t = time.perf_counter()
query = next(iter(model.embed_text(QUERIES)))
check(
    "canonical query (installed fastembed's placeholder)",
    float(np.abs(query[:7, :7] - CANONICAL_QUERY).max()),
    2e-3,
    time.perf_counter() - t,
)

t = time.perf_counter()
padded = list(model.embed_image([wide, image], batch_size=2))
alone = list(model.embed_image([wide], batch_size=1))[0]


def real_rows(
    x,
):  # padding positions are zeroed by the attention mask; padding may be on either side
    return x[np.abs(x).sum(-1) > 0]


diff = max(
    float(np.abs(real_rows(padded[1]) - real_rows(single)).max()),
    float(np.abs(real_rows(padded[0]) - real_rows(alone)).max()),
)
check("image in zero-padded batch == image alone", diff, 1e-4, time.perf_counter() - t)

session = ort.InferenceSession(
    str(Path(model_dir) / "model.onnx"), providers=["CPUExecutionProvider"]
)
encoded = model.model.tokenize(QUERIES)
ids = np.array([e.ids for e in encoded], dtype=np.int64)
mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
outs = {}
for tiles in (ids.shape[1], 1, 0):
    t = time.perf_counter()
    outs[tiles] = session.run(
        None,
        {
            "input_ids": ids,
            "attention_mask": mask,
            "pixel_values": np.zeros((len(ids), tiles, 3, 512, 512), np.float32),
        },
    )[0]
    print(f"  queries with {tiles} blank tiles per query: {time.perf_counter() - t:.2f}s")
check(
    "queries: 1 tile vs one per token", float(np.abs(outs[1] - outs[ids.shape[1]]).max()), 1e-5, 0
)
check(
    "queries: 0 tiles vs one per token", float(np.abs(outs[0] - outs[ids.shape[1]]).max()), 1e-5, 0
)

print("FAILED: " + ", ".join(failures) if failures else "ALL PASSED")
sys.exit(1 if failures else 0)
