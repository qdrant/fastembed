"""Qwen3 reranker using causal LM with yes/no logit scoring.

Unlike traditional cross-encoder rerankers (which concatenate query+document
as a pair, feed through a BERT-class model, and read a relevance head), the
Qwen3 reranker:

1. Formats input as a **chat template** with system/user/assistant turns.
2. Runs a **causal language model** (Qwen3ForCausalLM).
3. Extracts the **last-token logits** for the "yes" and "no" tokens.
4. Applies **softmax** to obtain the relevance probability.

This means the ONNX model output has shape ``(batch, seq_len, vocab_size)``
instead of the typical ``(batch, num_labels)`` from cross-encoders.
"""

from typing import Any

import numpy as np

from fastembed.common.model_description import BaseModelDescription, ModelSource
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.types import NumpyArray
from fastembed.rerank.cross_encoder.onnx_text_cross_encoder import (
    OnnxTextCrossEncoder,
    TextCrossEncoderWorker,
)
from fastembed.rerank.cross_encoder.onnx_text_model import TextRerankerWorker

# ---------------------------------------------------------------------------
# Qwen3 reranker constants
# ---------------------------------------------------------------------------
# Token IDs in the Qwen3 tokenizer vocabulary
TOKEN_YES_ID = 9693
TOKEN_NO_ID = 2132

SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)

DEFAULT_INSTRUCTION = (
    "Given a query and a document, judge whether the document is relevant to the query."
)

RERANK_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n<Instruct>: {instruction}\n"
    "<Query>: {query}\n<Document>: {document}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
supported_qwen3_reranker_models: list[BaseModelDescription] = [
    BaseModelDescription(
        model="Qwen/Qwen3-Reranker-0.6B",
        description=(
            "Qwen3 reranker (0.6B) using causal LM yes/no scoring. "
            "Multilingual, 40960 input tokens, instruction-aware, 2025 year."
        ),
        license="apache-2.0",
        size_in_GB=0.57,
        sources=ModelSource(hf="n24q02m/Qwen3-Reranker-0.6B-ONNX"),
        model_file="onnx/model.onnx",
    ),
]


# ---------------------------------------------------------------------------
# Qwen3 reranker implementation
# ---------------------------------------------------------------------------
class Qwen3CrossEncoder(OnnxTextCrossEncoder):
    """Qwen3 Reranker using causal LM with yes/no logit scoring.

    Usage::

        from fastembed import TextCrossEncoder

        reranker = TextCrossEncoder("Qwen/Qwen3-Reranker-0.6B")
        scores = list(reranker.rerank("What is AI?", ["doc1", "doc2"]))

        # Custom instruction
        scores = list(reranker.rerank(
            "What is AI?",
            ["doc1", "doc2"],
            instruction="Judge document relevance for code search.",
        ))
    """

    @classmethod
    def _list_supported_models(cls) -> list[BaseModelDescription]:
        return supported_qwen3_reranker_models

    # ------------------------------------------------------------------
    # Chat template formatting
    # ------------------------------------------------------------------
    @staticmethod
    def _format_rerank_input(
        query: str,
        document: str,
        instruction: str = DEFAULT_INSTRUCTION,
    ) -> str:
        """Build the chat-template string for a single query-document pair."""
        return RERANK_TEMPLATE.format(
            system=SYSTEM_PROMPT,
            instruction=instruction,
            query=query,
            document=document,
        )

    # ------------------------------------------------------------------
    # Yes/No logit scoring
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_yes_no_scores(model_output: NumpyArray) -> NumpyArray:
        """Extract yes/no logits from causal LM output and compute scores.

        Args:
            model_output: Raw model output, shape ``(batch, seq_len, vocab_size)``.

        Returns:
            Relevance scores (P(yes)), shape ``(batch,)``.
        """
        # Last token logits for each sample
        last_logits: NumpyArray = model_output[:, -1, :]  # (batch, vocab_size)

        # Stack [no, yes] logits
        yes_no_logits = np.stack(
            [last_logits[:, TOKEN_NO_ID], last_logits[:, TOKEN_YES_ID]], axis=1
        )  # (batch, 2)

        # Numerically stable softmax
        max_logits = np.max(yes_no_logits, axis=1, keepdims=True)
        exp_logits = np.exp(yes_no_logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return probs[:, 1]  # P(yes)

    # ------------------------------------------------------------------
    # Override ONNX inference to use chat-template + CausalLM scoring
    # ------------------------------------------------------------------
    def onnx_embed(self, query: str, documents: list[str], **kwargs: Any) -> OnnxOutputContext:
        """Score query-document pairs using the Qwen3 chat template."""
        instruction = kwargs.pop("instruction", DEFAULT_INSTRUCTION)
        texts = [self._format_rerank_input(query, doc, instruction) for doc in documents]
        return self._onnx_embed_texts(texts, **kwargs)

    def onnx_embed_pairs(self, pairs: list[tuple[str, str]], **kwargs: Any) -> OnnxOutputContext:
        """Score pre-formed (query, document) pairs."""
        instruction = kwargs.pop("instruction", DEFAULT_INSTRUCTION)
        texts = [self._format_rerank_input(query, doc, instruction) for query, doc in pairs]
        return self._onnx_embed_texts(texts, **kwargs)

    def _onnx_embed_texts(self, texts: list[str], **kwargs: Any) -> OnnxOutputContext:
        """Tokenise and run model one text at a time (static batch=1 ONNX graph),
        then concatenate the yes/no scores."""
        assert self.tokenizer is not None, "Tokenizer not loaded. Call load_onnx_model() first."

        all_scores: list[NumpyArray] = []
        for text in texts:
            tokenized = self.tokenizer.encode_batch([text])

            input_names: set[str] = {node.name for node in self.model.get_inputs()}  # type: ignore[union-attr]
            onnx_input: dict[str, NumpyArray] = {
                "input_ids": np.array([tokenized[0].ids], dtype=np.int64),
            }
            if "attention_mask" in input_names:
                onnx_input["attention_mask"] = np.array(
                    [tokenized[0].attention_mask], dtype=np.int64
                )
            if "token_type_ids" in input_names:
                onnx_input["token_type_ids"] = np.zeros_like(
                    onnx_input["input_ids"], dtype=np.int64
                )

            onnx_input = self._preprocess_onnx_input(onnx_input, **kwargs)
            outputs = self.model.run(self.ONNX_OUTPUT_NAMES, onnx_input)  # type: ignore[union-attr]
            scores = self._compute_yes_no_scores(outputs[0])
            all_scores.append(scores)

        concatenated = np.concatenate(all_scores).astype(np.float32)
        return OnnxOutputContext(model_output=concatenated)

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    @classmethod
    def _get_worker_class(cls) -> type[TextRerankerWorker]:
        return Qwen3CrossEncoderWorker


class Qwen3CrossEncoderWorker(TextCrossEncoderWorker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs: Any,
    ) -> OnnxTextCrossEncoder:
        return Qwen3CrossEncoder(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
