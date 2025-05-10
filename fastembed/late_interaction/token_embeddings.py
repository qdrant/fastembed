from typing import Union, Iterable, Optional, List, Dict, Any, Type

import numpy as np

from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.late_interaction.late_interaction_embedding_base import LateInteractionTextEmbeddingBase
from fastembed.text.onnx_embedding import OnnxTextEmbedding
from fastembed.text.onnx_text_model import TextEmbeddingWorker

supported_token_embeddings_models = [
    {
        "model": "jinaai/jina-embeddings-v2-small-en-tokens",
        "dim": 512,
        "description": "Text embeddings, Unimodal (text), English, 8192 input tokens truncation,"
                       " Prefixes for queries/documents: not necessary, 2023 year.",
        "license": "apache-2.0",
        "size_in_GB": 0.12,
        "sources": {"hf": "xenova/jina-embeddings-v2-small-en"},
        "model_file": "onnx/model.onnx",
    },
]


class TokenEmbeddingsModel(OnnxTextEmbedding, LateInteractionTextEmbeddingBase):
    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        """Lists the supported models.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the model information.
        """
        return supported_token_embeddings_models

    @classmethod
    def _get_worker_class(cls) -> Type[TextEmbeddingWorker]:
        return TokensEmbeddingWorker

    def _post_process_onnx_output(self, output: OnnxOutputContext) -> Iterable[np.ndarray]:
        # Size: (batch_size, sequence_length, hidden_size)
        embeddings = output.model_output
        # Size: (batch_size, sequence_length)
        masks = output.attention_mask

        # For each document we only select those embeddings that are not masked out

        for i in range(embeddings.shape[0]):
            yield embeddings[i, masks[i] == 1]

    def embed(
            self,
            documents: Union[str, Iterable[str]],
            batch_size: int = 256,
            parallel: Optional[int] = None,
            **kwargs,
    ) -> Iterable[np.ndarray]:
        yield from OnnxTextEmbedding.embed(self, documents, batch_size=batch_size, parallel=parallel, **kwargs)

    def tokenize_docs(self, documents: List[str]) -> List[np.ndarray]:
        encoded = self.tokenizer.encode_batch(documents)
        return [e.ids for e in encoded]


class TokensEmbeddingWorker(TextEmbeddingWorker):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs) -> TokenEmbeddingsModel:
        return TokenEmbeddingsModel(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )


if __name__ == "__main__":
    # Example usage
    model = TokenEmbeddingsModel(model_name="jinaai/jina-embeddings-v2-small-en-tokens")
    docs = ["Hello, world!", "hello", "hello hello"]

    embeddings = model.embed(docs)
    for emb in embeddings:
        print(emb.shape)

    print(model.tokenize_docs(docs))
