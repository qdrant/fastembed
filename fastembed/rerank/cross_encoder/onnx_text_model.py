from typing import Sequence, Optional, List, Dict, Iterable
from pathlib import Path

import numpy as np
from tokenizers import Encoding

from fastembed.common.onnx_model import OnnxModel, OnnxProvider, OnnxOutputContext
from fastembed.common.preprocessor_utils import load_tokenizer
from fastembed.common.utils import iter_batch


class OnnxCrossEncoderModel(OnnxModel):
    ONNX_OUTPUT_NAMES: Optional[List[str]] = None

    def _load_onnx_model(
        self,
        model_dir: Path,
        model_file: str,
        threads: Optional[int],
        providers: Optional[Sequence[OnnxProvider]] = None,
        cuda: bool = False,
        device_id: Optional[int] = None,
    ) -> None:
        super()._load_onnx_model(
            model_dir=model_dir,
            model_file=model_file,
            threads=threads,
            providers=providers,
            cuda=cuda,
            device_id=device_id,
        )
        self.tokenizer, _ = load_tokenizer(model_dir=model_dir)

    def tokenize(self, query: str, documents: List[str], **kwargs) -> List[Encoding]:
        return self.tokenizer.encode_batch([(query, doc) for doc in documents])

    def onnx_embed(self, query: str, documents: List[str], **kwargs) -> OnnxOutputContext:
        tokenized_input = self.tokenize(query, documents, **kwargs)

        inputs = {
            "input_ids": np.array([enc.ids for enc in tokenized_input], dtype=np.int64),
            "attention_mask": np.array(
                [enc.attention_mask for enc in tokenized_input], dtype=np.int64
            ),
        }
        input_names = {node.name for node in self.model.get_inputs()}
        if "token_type_ids" in input_names:
            inputs["token_type_ids"] = np.array(
                [enc.type_ids for enc in tokenized_input], dtype=np.int64
            )

        onnx_input = self._preprocess_onnx_input(inputs, **kwargs)
        outputs = self.model.run(self.ONNX_OUTPUT_NAMES, onnx_input)
        return OnnxOutputContext(model_output=outputs[0][:, 0].tolist())

    def _rerank_documents(
        self, query: str, documents: Iterable[str], batch_size: int, **kwargs
    ) -> Iterable[float]:
        if not hasattr(self, "model") or self.model is None:
            self.load_onnx_model()
        for batch in iter_batch(documents, batch_size):
            yield from self.onnx_embed(query, batch, **kwargs).model_output

    def _preprocess_onnx_input(
        self, onnx_input: Dict[str, np.ndarray], **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Preprocess the onnx input.
        """
        return onnx_input
