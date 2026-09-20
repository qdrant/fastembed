# Synthetic token lookup fixture

`token_lookup.onnx` is an original synthetic graph for offline process-parallel
embedding tests. It contains no trained weights or external model data.

The graph is one `Gather` node (axis 0), with an int64 `input_ids` input of shape
`[batch, sequence]` and float32 output `[batch, sequence, 2]`. Its constant table is:

| Token ID | Vector |
| --- | --- |
| 0 (unknown) | [0, 0] |
| 1 (padding) | [0, 0] |
| 2 (alpha) | [2, 4] |
| 3 (beta) | [3, 9] |
| 4 (gamma) | [4, 16] |

This makes CLS, mean and last-token pooling expectations directly calculable.
Tests generate their tokenizer metadata in a temporary directory and run actual
ONNX Runtime inference. The fixture uses ONNX opset 13 and IR version 8.

Regenerate with the development dependencies installed:

```sh
python tests/misc/generate_token_lookup.py
```

Generated with ONNX 1.23.0; size 234 bytes; SHA-256:

```text
8ff79ae726a3cb5cad3c6b4f6c458dc2e085f815c13f6117ba4df106911c26bf
```

The generator also runs `onnx.checker.check_model`. Normal pytest execution loads
the checked-in file through ONNX Runtime and does not import the development-only
`onnx` package. The builtin-identity control deliberately uses this synthetic
graph too; it tests dispatch/inference parity, not pretrained model quality.
