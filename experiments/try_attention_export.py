import numpy as np
import onnx
import onnxruntime
from transformers import AutoTokenizer

model_id = "sentence-transformers/paraphrase-MiniLM-L6-v2"
output_dir = f"models/{model_id.replace('/', '_')}"
model_kwargs = {"output_attentions": True, "return_dict": True}
tokenizer = AutoTokenizer.from_pretrained(model_id)

model_path = f"{output_dir}/model.onnx"
onnx_model = onnx.load(model_path)
ort_session = onnxruntime.InferenceSession(model_path)
text = "This is a test sentence"
tokenizer_output = tokenizer(text, return_tensors="np")
input_ids = tokenizer_output["input_ids"]
attention_mask = tokenizer_output["attention_mask"]
print(attention_mask)
# Prepare the input
input_ids = np.array(input_ids).astype(np.int64)  # Replace your_input_ids with actual input data

# Run the ONNX model
outputs = ort_session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})

# Get the attention weights
attentions = outputs[-1]

# Print the attention weights for the first layer and first head
print(attentions[0][0])
