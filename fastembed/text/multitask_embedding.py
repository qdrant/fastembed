from enum import Enum
from typing import Any, Type, Iterable, Union, Optional

import numpy as np

from fastembed.common.types import NumpyArray
from fastembed.text.pooled_normalized_embedding import PooledNormalizedEmbedding
from fastembed.text.onnx_embedding import OnnxTextEmbeddingWorker
from fastembed.common.model_description import DenseModelDescription, ModelSource

supported_multitask_models: list[DenseModelDescription] = [
    DenseModelDescription(
        model="jinaai/jina-embeddings-v3",
        dim=1024,
        tasks={
            "retrieval.query": 0,
            "retrieval.passage": 1,
            "separation": 2,
            "classification": 3,
            "text-matching": 4,
        },
        description=(
            "Multi-task unimodal (text) embedding model, multi-lingual (~100), "
            "1024 tokens truncation, and 8192 sequence length. Prefixes for queries/documents: not necessary, 2024 year."
        ),
        license="cc-by-nc-4.0",
        size_in_GB=2.29,
        sources=ModelSource(hf="jinaai/jina-embeddings-v3"),
        model_file="onnx/model.onnx",
        additional_files=["onnx/model.onnx_data"],
    ),
]


class Task(int, Enum):
    RETRIEVAL_QUERY = 0
    RETRIEVAL_PASSAGE = 1
    SEPARATION = 2
    CLASSIFICATION = 3
    TEXT_MATCHING = 4


class JinaEmbeddingV3(PooledNormalizedEmbedding):
    PASSAGE_TASK = Task.RETRIEVAL_PASSAGE
    QUERY_TASK = Task.RETRIEVAL_QUERY

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.current_task_id: Union[Task, int] = self.PASSAGE_TASK

    @classmethod
    def _get_worker_class(cls) -> Type[OnnxTextEmbeddingWorker]:
        return JinaEmbeddingV3Worker

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        return supported_multitask_models

    def _preprocess_onnx_input(
        self, onnx_input: dict[str, NumpyArray], **kwargs: Any
    ) -> dict[str, NumpyArray]:
        onnx_input["task_id"] = np.array(self.current_task_id, dtype=np.int64)
        return onnx_input

    def embed(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        task_id: int = PASSAGE_TASK,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        self.current_task_id = task_id
        kwargs["task_id"] = task_id
        yield from super().embed(documents, batch_size, parallel, **kwargs)

    def query_embed(self, query: Union[str, Iterable[str]], **kwargs: Any) -> Iterable[NumpyArray]:
        self.current_task_id = self.QUERY_TASK
        yield from super().embed(query, **kwargs)

    def passage_embed(self, texts: Iterable[str], **kwargs: Any) -> Iterable[NumpyArray]:
        self.current_task_id = self.PASSAGE_TASK
        yield from super().embed(texts, **kwargs)


class JinaEmbeddingV3Worker(OnnxTextEmbeddingWorker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs: Any,
    ) -> JinaEmbeddingV3:
        model = JinaEmbeddingV3(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
        model.current_task_id = kwargs["task_id"]
        return model
