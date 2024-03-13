from optimum.exporters.onnx import main_export
from transformers import AutoTokenizer

model_id = "sentence-transformers/paraphrase-MiniLM-L6-v2"
output_dir = f"models/{model_id.replace('/', '_')}"
model_kwargs = {"output_attentions": True, "return_dict": True}
tokenizer = AutoTokenizer.from_pretrained(model_id)

# export if the output model does not exist
# try:
#     sess = onnxruntime.InferenceSession(f"{output_dir}/model.onnx")
#     print("Model already exported")
# except FileNotFoundError:
print(f"Exporting model to {output_dir}")
main_export(model_id, output=output_dir, no_post_process=True, model_kwargs=model_kwargs)
