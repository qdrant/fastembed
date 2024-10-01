from typing import Sequence, Optional
from pathlib import Path

from fastembed.common.onnx_model import OnnxModel, OnnxProvider
from fastembed.common.preprocessor_utils import load_tokenizer


class OnnxCrossEncoderModel(OnnxModel):
    def load_onnx_model(
        self,
        model_dir: Path,
        model_file: str,
        threads: Optional[int],
        providers: Optional[Sequence[OnnxProvider]] = None,
    ) -> None:
        super().load_onnx_model(
            model_dir=model_dir,
            model_file=model_file,
            threads=threads,
            providers=providers,
        )
        self.tokenizer, _ = load_tokenizer(model_dir=model_dir)

    def onnx_embed(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        tokenized_input = self.tokenizer.encode_batch([(query, doc) for doc in documents])

        inputs = {
            "input_ids": [enc.ids for enc in tokenized_input],
            "attention_mask": [enc.attention_mask for enc in tokenized_input],
        }

        for input_name in self.model.get_inputs():
            if input_name.name == "token_type_ids":
                inputs["token_type_ids"] = [enc.type_ids for enc in tokenized_input]

        outputs = self.model.run(self.ONNX_OUTPUT_NAMES, inputs)

        return outputs[0][:, 0].tolist()
