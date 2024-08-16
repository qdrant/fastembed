from typing import Sequence, Optional
from pathlib import Path

from onnxruntime import InferenceSession
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

from fastembed.common.onnx_model import OnnxModel, OnnxProvider
from fastembed.common.preprocessor_utils import load_tokenizer


class OnnxCrossEncoderModel(OnnxModel):
    def __init__(
        self,
        model_name: str,
        cache_dir: str = None,
        threads: int = None,
        providers: Sequence[OnnxProvider] = None,
        **kwargs,
    ):
        self.tokenizer: Tokenizer = None
        self.model: InferenceSession = None
        self.model_name: str = model_name
        self.cache_dir: str = cache_dir
        self.threads: int = threads
        self.providers: Sequence[OnnxProvider] = providers

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

    def configure_tokenizer(self) -> None:
        """Configures the tokenizer to properly handle query and document pairs."""
        cls_token_id = self.tokenizer.token_to_id("[CLS]")
        sep_token_id = self.tokenizer.token_to_id("[SEP]")

        if cls_token_id is not None and sep_token_id is not None:
            self.tokenizer.post_processor = TemplateProcessing(
                single="[CLS] $A [SEP]",
                pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                special_tokens=[
                    ("[CLS]", cls_token_id),
                    ("[SEP]", sep_token_id),
                ],
            )
        else:
            print("Tokenizer does not have [CLS] or [SEP] tokens. Skipping post-processing setup.")

    def onnx_embed(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        tokenized_input = self.tokenizer.encode_batch([(query, doc) for doc in documents])

        inputs = {
            "input_ids": [enc.ids for enc in tokenized_input],
            "attention_mask": [enc.attention_mask for enc in tokenized_input],
        }

        if "token_type_ids" in [input.name for input in self.model.get_inputs()]:
            inputs["token_type_ids"] = [enc.type_ids for enc in tokenized_input]

        outputs = self.model.run(None, inputs)

        return outputs[0][:, 0].tolist()

    def get_tokenizer(self):
        return self.tokenizer
