from typing import List, Union, Iterable, Dict, Any, Sequence, Optional
from fastembed.common import OnnxProvider
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.rerank.cross_encoder.onnx_text_model import OnnxCrossEncoderModel
from fastembed.rerank.cross_encoder.text_cross_encoder_base import TextCrossEncoderBase
import numpy as np

supported_onnx_models = [
    {
        "model": "Xenova/ms-marco-MiniLM-L-6-v2",
        "dim": 1024,
        "sources": {
            "hf": "Xenova/ms-marco-MiniLM-L-6-v2",
        },
        "model_file": "onnx/model.onnx"
    },
    {
        "model": "Xenova/ms-marco-MiniLM-L-12-v2",
        "dim": 1024,
        "sources": {
            "hf": "Xenova/ms-marco-MiniLM-L-12-v2",
        },
        "model_file": "onnx/model.onnx"
    },
    {
        "model": "BAAI/bge-reranker-base",
        "dim": 1024,
        "sources": {
            "hf": "BAAI/bge-reranker-base",
        },
        "model_file": "onnx/model.onnx"
    }
]

class OnnxTextCrossEncoder(TextCrossEncoderBase, OnnxCrossEncoderModel):
    ONNX_OUTPUT_NAMES = ["logits"]

    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        return supported_onnx_models

    def __init__(
            self,
            model_name: str, 
            cache_dir: Optional[str] = None, 
            threads: Optional[int] = None, 
            providers: Optional[Sequence[OnnxProvider]] = None, 
            **kwargs
    ):
        TextCrossEncoderBase.__init__(self, model_name, cache_dir, threads, **kwargs)
        OnnxCrossEncoderModel.__init__(self, model_name, cache_dir, threads, providers, **kwargs)
        self.load_onnx_model(self.model_name, self.cache_dir, self.threads, providers)
        self.tokenizer = self.get_tokenizer()

    def rerank(
            self, 
            query: str, 
            documents: Union[str, Iterable[str]], 
            **kwargs
    ) -> List[float]:
        if not documents:
            return []
        inputs = self._prepare_inputs(query, documents)
        outputs = self._run_model(inputs)
        scores = self._post_process(outputs)
        return scores

    def _prepare_inputs(
            self, query: str, documents: Iterable[str]
    ) -> Dict[str, np.ndarray]:
        inputs = self.tokenizer([query] * len(documents), list(documents), padding=True, truncation=True, return_tensors="np")
        return inputs

    def _run_model(self, inputs: Dict[str, np.ndarray]) -> OnnxOutputContext:
        return self.model.run(self.ONNX_OUTPUT_NAMES, inputs)

    def _post_process(self, outputs: OnnxOutputContext) -> List[float]:
        return outputs[0].tolist()
