from typing import Sequence, Dict, List

import numpy as np

from fastembed.common.onnx_model import OnnxModel, OnnxProvider
from onnxruntime import InferenceSession
from tokenizers import Tokenizer, Encoding
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

class OnnxCrossEncoderModel(OnnxModel):
    def __init__(
            self, 
            model_name: str, 
            cache_dir: str = None, 
            threads: int = None, 
            providers: Sequence[OnnxProvider] = None, 
            **kwargs):
        super().__init__()
        self.tokenizer = self._load_tokenizer(model_name, cache_dir)
        self.model = self._load_model(model_name, cache_dir, providers)

    def _load_tokenizer(self, model_name: str, cache_dir: str = None) -> Tokenizer:
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", 1),
                ("[SEP]", 2),
            ],
        )
        return tokenizer
    
    def _load_model(self, model_name: str, cache_dir: str = None, providers: Sequence[OnnxProvider] = None) -> InferenceSession:
        model_path = f"{cache_dir}/{model_name}.onnx"
        session_options = None
        if threads:
            session_options = InferenceSession.SessionOptions()
            session_options.intra_op_num_threads = threads
        return InferenceSession(model_path, providers=providers, sess_options=session_options)

    def onnx_embed(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        tokenized_input = self.tokenizer.encode_batch([(query, doc) for doc in documents])
        
        inputs = {
            "input_ids": [enc.ids for enc in tokenized_input],
            "attention_mask": [enc.attention_mask for enc in tokenized_input]
        }
        
        outputs = self.model.run(None, inputs)
        
        return outputs[0][:, 0].tolist()

    def get_tokenizer(self):
        return self.tokenizer