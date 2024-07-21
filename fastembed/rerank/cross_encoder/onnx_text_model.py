from typing import Sequence
from fastembed.common.onnx_model import OnnxModel, OnnxProvider
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

class OnnxCrossEncoderModel(OnnxModel):
    def __init__(
            self, 
            model_name: str, 
            cache_dir: str = None, 
            threads: int = None, 
            providers: Sequence[OnnxProvider] = None, 
    **kwargs):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = ORTModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir, providers=providers)

    def get_tokenizer(self):
        return self.tokenizer
    
    def encode(
            self, 
            query: str, 
            documents: Sequence[str]
    ) -> Sequence[float]:
        inputs = self.tokenizer([query] * len(documents), documents, padding=True, truncation=True, return_tensors="np")
        outputs = self.model(**inputs)
        return outputs.logits[:, 0].tolist()
