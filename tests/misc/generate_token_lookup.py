"""Regenerate token_lookup.onnx; requires the development-only onnx package."""

from pathlib import Path

import onnx
from onnx import TensorProto, helper

table = helper.make_tensor("table", TensorProto.FLOAT, [5, 2], [0, 0, 0, 0, 2, 4, 3, 9, 4, 16])
graph = helper.make_graph(
    [helper.make_node("Gather", ["table", "input_ids"], ["embeddings"], axis=0)],
    "synthetic-token-lookup",
    [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "sequence"])],
    [helper.make_tensor_value_info("embeddings", TensorProto.FLOAT, ["batch", "sequence", 2])],
    [table],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, Path(__file__).with_name("token_lookup.onnx"))
