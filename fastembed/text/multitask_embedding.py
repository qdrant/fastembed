from enum import Enum
from typing import Any, Type, Iterable, Union, Optional

import numpy as np

from fastembed.text.pooled_normalized_embedding import PooledNormalizedEmbedding
from fastembed.text.onnx_embedding import OnnxTextEmbeddingWorker
from fastembed.text.onnx_text_model import TextEmbeddingWorker

supported_multitask_models = [
    {
        "model": "jinaai/jina-embeddings-v3",
        "dim": 1024,
        "tasks": {
            "retrieval.query": 0,
            "retrieval.passage": 1,
            "separation": 2,
            "classification": 3,
            "text-matching": 4,
        },
        "description": "Multi-task, multi-lingual embedding model with Matryoshka architecture",
        "license": "cc-by-nc-4.0",
        "size_in_GB": 2.29,
        "sources": {
            "hf": "jinaai/jina-embeddings-v3",
        },
        "model_file": "onnx/model.onnx",
        "additional_files": ["onnx/model.onnx_data"],
    },
]


class Task(int, Enum):
    RETRIEVAL_QUERY = 0
    RETRIEVAL_PASSAGE = 1
    SEPARATION = 2
    CLASSIFICATION = 3
    TEXT_MATCHING = 4


class JinaEmbeddingV3(PooledNormalizedEmbedding):
    DEFAULT_TASK = Task.TEXT_MATCHING
    PASSAGE_TASK = Task.RETRIEVAL_PASSAGE
    QUERY_TASK = Task.RETRIEVAL_QUERY

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_task_id = self.DEFAULT_TASK

    @classmethod
    def _get_worker_class(cls) -> Type["TextEmbeddingWorker"]:
        return JinaEmbeddingV3Worker

    @classmethod
    def list_supported_models(cls) -> list[dict[str, Any]]:
        return supported_multitask_models

    def _preprocess_onnx_input(
        self, onnx_input: dict[str, np.ndarray], **kwargs
    ) -> dict[str, np.ndarray]:
        onnx_input["task_id"] = np.array(self._current_task_id, dtype=np.int64)
        return onnx_input

    def embed(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        task_id: int = 4,
        **kwargs,
    ) -> Iterable[np.ndarray]:
        self._current_task_id = task_id
        yield from super().embed(documents, batch_size, parallel, **kwargs)

    def query_embed(self, query: Union[str, Iterable[str]], **kwargs) -> Iterable[np.ndarray]:
        self._current_task_id = self.QUERY_TASK

        if isinstance(query, str):
            query = [query]

        if not hasattr(self, "model") or self.model is None:
            self.load_onnx_model()

        for text in query:
            yield from self._post_process_onnx_output(self.onnx_embed([text]))

    def passage_embed(self, texts: Iterable[str], **kwargs) -> Iterable[np.ndarray]:
        self._current_task_id = self.PASSAGE_TASK

        if not hasattr(self, "model") or self.model is None:
            self.load_onnx_model()

        for text in texts:
            yield from self._post_process_onnx_output(self.onnx_embed([text]))


class JinaEmbeddingV3Worker(OnnxTextEmbeddingWorker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs,
    ) -> JinaEmbeddingV3:
        return JinaEmbeddingV3(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
