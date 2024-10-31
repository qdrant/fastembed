from typing import Any, Dict, List, Type

import numpy as np
from tokenizers import Encoding

from fastembed.late_interaction.colbert import Colbert
from fastembed.text.onnx_text_model import TextEmbeddingWorker

supported_jina_colbert_models = [
    {
        "model": "jinaai/jina-colbert-v2",
        "dim": 1024,
        "description": "New model that expands capabilities of colbert-v1 with multilingual and context length of 8192, 2024 year",
        "license": "cc-by-nc-4.0",
        "size_in_GB": 2.24,
        "sources": {
            "hf": "jinaai/jina-colbert-v2",
        },
        "model_file": "onnx/model.onnx",
        "additional_files": ["onnx/model.onnx_data"],
    },
]


class JinaColbert(Colbert):
    QUERY_MARKER_TOKEN_ID = 250002
    DOCUMENT_MARKER_TOKEN_ID = 250003
    MIN_QUERY_LENGTH = 32
    MASK_TOKEN = "<mask>"

    @classmethod
    def _get_worker_class(cls) -> Type[TextEmbeddingWorker]:
        return JinaColbertEmbeddingWorker

    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        """Lists the supported models.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the model information.
        """
        return supported_jina_colbert_models

    def _preprocess_onnx_input(
        self, onnx_input: Dict[str, np.ndarray], is_doc: bool = True
    ) -> Dict[str, np.ndarray]:
        if is_doc:
            onnx_input["input_ids"][:, 1] = self.DOCUMENT_MARKER_TOKEN_ID
        else:
            onnx_input["input_ids"][:, 1] = self.QUERY_MARKER_TOKEN_ID
        return onnx_input

    def _tokenize_query(self, query: str) -> List[Encoding]:
        # "@ " is added to a query to be replaced with a special query token
        # "@ " is considered as one token in jina-colbert-v2 tokenizer
        query = f"@ {query}"
        encoded = self.tokenizer.encode_batch([query])
        # colbert authors recommend to pad queries with [MASK] tokens for query augmentation to improve performance
        if len(encoded[0].ids) < self.MIN_QUERY_LENGTH:
            prev_padding = None
            if self.tokenizer.padding:
                prev_padding = self.tokenizer.padding
            self.tokenizer.enable_padding(
                pad_token=self.MASK_TOKEN,
                pad_id=self.mask_token_id,
                length=self.MIN_QUERY_LENGTH,
            )
            encoded = self.tokenizer.encode_batch([query])
            if prev_padding is None:
                self.tokenizer.no_padding()
            else:
                self.tokenizer.enable_padding(**prev_padding)
        # the attention mask for jina-colbert-v2 is always 1 in queries
        encoded["attention_mask"][:] = 1
        return encoded

    def _tokenize_documents(self, documents: List[str]) -> List[Encoding]:
        # "@ " is added to a document to be replaced with a special document token
        # "@ " is considered as one token in jina-colbert-v2 tokenizer
        documents = ["@ " + doc for doc in documents]
        encoded = self.tokenizer.encode_batch(documents)
        return encoded


class JinaColbertEmbeddingWorker(TextEmbeddingWorker):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs) -> JinaColbert:
        return JinaColbert(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
