from typing import Sequence, Optional
from pathlib import Path

from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing

from fastembed.common.onnx_model import OnnxModel, OnnxProvider
from fastembed.common.preprocessor_utils import load_tokenizer


class OnnxCrossEncoderModel(OnnxModel):
    ONNX_INPUT_NAMES = None
    ONNX_OUTPUT_NAMES = None

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
        self.configure_tokenizer()
        self.ONNX_INPUT_NAMES = [input_meta.name for input_meta in self.model.get_inputs()]

    def configure_tokenizer(self) -> None:
        """Configures the tokenizer to properly handle query and document pairs."""
        cls_token_id = self.tokenizer.token_to_id("[CLS]") or self.tokenizer.token_to_id("<s>")
        sep_token_id = self.tokenizer.token_to_id("[SEP]") or self.tokenizer.token_to_id("</s>")

        if cls_token_id is None or sep_token_id is None:
            raise ValueError("Tokenizer does not have [CLS] or [SEP] tokens, or their equivalent.")

        self.tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", cls_token_id),
                ("[SEP]", sep_token_id),
            ],
        )

    def onnx_embed(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        if self.model_name == "BAAI/bge-reranker-base":
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            tokenized_input = tokenizer(
                [(query, doc) for doc in documents],
                padding=True,
                truncation=True,
                return_tensors="np",
                max_length=512,
            )
            inputs = {
                "input_ids": tokenized_input["input_ids"],
                "attention_mask": tokenized_input["attention_mask"],
            }
            if "token_type_ids" in self.ONNX_INPUT_NAMES:
                inputs["token_type_ids"] = tokenized_input["token_type_ids"]
        else:
            tokenized_input = self.tokenizer.encode_batch([(query, doc) for doc in documents])

            inputs = {
                "input_ids": [enc.ids for enc in tokenized_input],
                "attention_mask": [enc.attention_mask for enc in tokenized_input],
            }
            if "token_type_ids" in self.ONNX_INPUT_NAMES:
                inputs["token_type_ids"] = [enc.type_ids for enc in tokenized_input]

        outputs = self.model.run(self.ONNX_OUTPUT_NAMES, inputs)

        return outputs[0][:, 0].tolist()
