"""Export an inference-free SPLADE document encoder to ONNX.

Converts `opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte` (an MLM head
over a GTE backbone) into an onnx model producing token logits, and assembles a model dir
with everything fastembed's `IfSplade` needs: model.onnx, tokenizer files and idf.json.

Usage:
    python experiments/if_splade_to_onnx.py --output-dir models/opensearch-neural-sparse-encoding-doc-v3-gte
"""

import argparse
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_ID = "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte"
# revision of the remote modeling code (Alibaba-NLP/new-impl), pinned in the model card
CODE_REVISION = "40ced75c3017eb27626c9d4ea981bde21a2662f4"

TOKENIZER_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "idf.json",
]


class LogitsOnly(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def export(model_id: str, output_dir: Path, opset: int = 14) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForMaskedLM.from_pretrained(
        model_id, trust_remote_code=True, code_revision=CODE_REVISION
    )
    model.eval()
    wrapped = LogitsOnly(model)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dummy = tokenizer(
        ["fastembed is a library", "onnx export"],
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=False,
    )

    onnx_path = output_dir / "model.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            wrapped,
            (dummy["input_ids"], dummy["attention_mask"]),
            f=onnx_path.as_posix(),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
            do_constant_folding=True,
            opset_version=opset,
            dynamo=False,
        )

    for file_name in TOKENIZER_FILES:
        local_path = hf_hub_download(repo_id=model_id, filename=file_name)
        shutil.copy(local_path, output_dir / file_name)

    return onnx_path


def parity_check(model_id: str, output_dir: Path) -> None:
    import numpy as np
    import onnxruntime as ort

    model = AutoModelForMaskedLM.from_pretrained(
        model_id, trust_remote_code=True, code_revision=CODE_REVISION
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    documents = [
        "Currently New York is rainy.",
        "fastembed is a lightweight library for generating embeddings",
        "hello world",
    ]
    features = tokenizer(
        documents, padding=True, truncation=True, return_tensors="pt", return_token_type_ids=False
    )

    with torch.inference_mode():
        torch_logits = model(**features).logits.numpy()

    session = ort.InferenceSession(output_dir / "model.onnx")
    onnx_logits = session.run(
        ["logits"],
        {
            "input_ids": features["input_ids"].numpy(),
            "attention_mask": features["attention_mask"].numpy(),
        },
    )[0]

    max_diff = np.abs(torch_logits - onnx_logits).max()
    print(f"max |torch - onnx| logits diff: {max_diff}")
    assert max_diff < 1e-3, "onnx export does not match the torch model"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-dir", default=f"models/{MODEL_ID.replace('/', '_')}", type=Path)
    parser.add_argument("--opset", default=14, type=int)
    args = parser.parse_args()

    onnx_path = export(args.model_id, args.output_dir, args.opset)
    print(f"Exported to {onnx_path}")
    parity_check(args.model_id, args.output_dir)


if __name__ == "__main__":
    main()
