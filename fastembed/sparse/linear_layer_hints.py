import onnx
from onnx import helper

# Load the original ONNX model
model = onnx.load('path_to_your_original_model.onnx')

# Assuming 'colbert_linear' and 'sparse_linear' are the names of the intermediate nodes you're interested in
# You need to find out the exact names by inspecting the model, e.g., using Netron

# Add these nodes as additional outputs to the model
output_for_colbert_linear = helper.make_tensor_value_info('colbert_linear', onnx.TensorProto.FLOAT, [your_shape_here])
output_for_sparse_linear = helper.make_tensor_value_info('sparse_linear', onnx.TensorProto.FLOAT, [your_shape_here])

model.graph.output.extend([output_for_colbert_linear, output_for_sparse_linear])

# Save the modified model
onnx.save(model, 'path_to_your_modified_model.onnx')
