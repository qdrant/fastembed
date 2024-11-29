import torch
from colpali_engine.models import ColPali, ColPaliProcessor
import onnxruntime as ort

model_name = "vidore/colpali-v1.2"
original_model = ColPali.from_pretrained(model_name).eval()
processor = ColPaliProcessor.from_pretrained(model_name)

dummy_query = ["Is attention really all you need?"]

# Process the input query
processed_query = processor.process_queries(dummy_query).to(original_model.device)

# Prepare input tensors
input_query_tensor = processed_query["input_ids"].type(torch.long)
attention_mask_tensor = processed_query["attention_mask"].type(torch.long)

# Export the model to ONNX with the required inputs and dynamic shapes
torch.onnx.export(
    original_model.model.language_model,
    (input_query_tensor, attention_mask_tensor),
    "experiments/colpali_text_encoder_dir/model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamo=True,
    opset_version=14,
)


image_session = ort.InferenceSession("experiments/colpali_text_encoder_dir/model.onnx")
print("Session output", image_session((input_query_tensor, attention_mask_tensor)))
