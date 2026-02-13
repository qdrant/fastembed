"""Qwen3 text embedding with last-token pooling and Matryoshka (MRL) support.

Qwen3-Embedding uses a causal LM architecture with last-token pooling instead
of the traditional CLS or mean pooling used by BERT-family models. It also
supports Matryoshka Representation Learning (MRL), allowing truncation of
embeddings to smaller dimensions (32-1024) with graceful degradation.

Key differences from standard text embedding models:
  - Last-token pooling: embedding is extracted from the last non-padding token
  - Left-padding: the tokenizer pads from the left (not right)
  - Instruction-aware: queries use ``Instruct: {task}\\nQuery: {text}`` format
  - MRL: pass ``dim=256`` (or any value 32-1024) to ``embed()`` / ``query_embed()``
"""

from collections.abc import Iterable
from typing import Any

from fastembed.common.model_description import DenseModelDescription, ModelSource
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.types import NumpyArray
from fastembed.common.utils import last_token_pool, normalize
from fastembed.text.onnx_embedding import OnnxTextEmbedding, OnnxTextEmbeddingWorker

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
supported_qwen3_models: list[DenseModelDescription] = [
    DenseModelDescription(
        model="Qwen/Qwen3-Embedding-0.6B",
        dim=1024,
        description=(
            "Qwen3 text embedding (0.6B) with last-token pooling and MRL support "
            "(32-1024 dims). Multilingual, 32768 input tokens, instruction-aware, "
            "2025 year."
        ),
        license="apache-2.0",
        size_in_GB=0.57,
        sources=ModelSource(hf="n24q02m/Qwen3-Embedding-0.6B-ONNX"),
        model_file="onnx/model.onnx",
    ),
]

# ---------------------------------------------------------------------------
# Instruction template
# ---------------------------------------------------------------------------
DEFAULT_TASK = "Given a query, retrieve relevant documents that answer the query"
QUERY_INSTRUCTION_TEMPLATE = "Instruct: {task}\nQuery: {text}"


# ---------------------------------------------------------------------------
# Qwen3 embedding implementation
# ---------------------------------------------------------------------------
class Qwen3TextEmbedding(OnnxTextEmbedding):
    """Qwen3 Embedding model with last-token pooling and MRL support.

    Usage::

        from fastembed import TextEmbedding

        model = TextEmbedding("Qwen/Qwen3-Embedding-0.6B")
        embeddings = list(model.embed(["Hello world"]))

        # MRL: reduce dimension
        embeddings_256 = list(model.embed(["Hello world"], dim=256))

        # Query with custom task instruction
        query_emb = list(model.query_embed("What is AI?", task="..."))
    """

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        return supported_qwen3_models

    def _post_process_onnx_output(
        self, output: OnnxOutputContext, **kwargs: Any
    ) -> Iterable[NumpyArray]:
        if output.attention_mask is None:
            raise ValueError("attention_mask must be provided for last-token pooling")

        embeddings = last_token_pool(output.model_output, output.attention_mask)

        # MRL: optionally truncate to requested dimension
        dim: int | None = kwargs.get("dim")
        if dim is not None:
            embeddings = embeddings[:, :dim]

        return normalize(embeddings)

    # ------------------------------------------------------------------
    # embed / query_embed / passage_embed
    # ------------------------------------------------------------------
    def embed(
        self,
        documents: str | Iterable[str],
        batch_size: int = 1,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        """Encode documents into embeddings.

        Args:
            documents: A single document string or an iterable of documents.
            batch_size: Ignored -- always ``1`` because
                the causal-LM ONNX graph has a static batch dimension.
            parallel: Number of parallel workers (``None`` = single-threaded).
            **kwargs: Extra arguments; ``dim`` (int) enables MRL truncation,
                ``task`` (str) is used only by :meth:`query_embed`.

        Yields:
            NumpyArray: L2-normalised embeddings, one per document.
        """
        yield from self._embed_documents(
            model_name=self.model_name,
            cache_dir=str(self.cache_dir),
            documents=documents,
            batch_size=1,
            parallel=parallel,
            providers=self.providers,
            cuda=self.cuda,
            device_ids=self.device_ids,
            local_files_only=self._local_files_only,
            specific_model_path=self._specific_model_path,
            extra_session_options=self._extra_session_options,
            **kwargs,
        )

    def query_embed(self, query: str | Iterable[str], **kwargs: Any) -> Iterable[NumpyArray]:
        """Embed queries with instruction prefix.

        The instruction prefix follows the Qwen3 format::

            Instruct: {task}
            Query: {query_text}

        Args:
            query: A single query string or an iterable of queries.
            **kwargs: ``task`` (str) overrides the default retrieval instruction.
                ``dim`` (int) enables MRL truncation.

        Yields:
            NumpyArray: L2-normalised query embeddings.
        """
        task = kwargs.pop("task", DEFAULT_TASK)
        if isinstance(query, str):
            queries = [QUERY_INSTRUCTION_TEMPLATE.format(task=task, text=query)]
        else:
            queries = [QUERY_INSTRUCTION_TEMPLATE.format(task=task, text=q) for q in query]
        yield from self.embed(queries, **kwargs)

    def passage_embed(self, texts: Iterable[str], **kwargs: Any) -> Iterable[NumpyArray]:
        """Embed passages (documents) without instruction prefix.

        Args:
            texts: An iterable of passage strings.
            **kwargs: ``dim`` (int) enables MRL truncation.

        Yields:
            NumpyArray: L2-normalised passage embeddings.
        """
        yield from self.embed(texts, **kwargs)

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    @classmethod
    def _get_worker_class(cls) -> type[OnnxTextEmbeddingWorker]:
        return Qwen3TextEmbeddingWorker


class Qwen3TextEmbeddingWorker(OnnxTextEmbeddingWorker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs: Any,
    ) -> OnnxTextEmbedding:
        return Qwen3TextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
