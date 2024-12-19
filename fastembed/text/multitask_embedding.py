from typing import Any, Type, Iterable, Union, Optional

import numpy as np

from fastembed.text.pooled_normalized_embedding import PooledNormalizedEmbedding
from fastembed.text.onnx_embedding import OnnxTextEmbeddingWorker
from fastembed.text.onnx_text_model import TextEmbeddingWorker

supported_multitask_models = [
    {
        "model": "jinaai/jina-embeddings-v3",
        "dim": [32, 64, 128, 256, 512, 768, 1024],
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


class JinaEmbeddingV3(PooledNormalizedEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_task_id = 4

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
        self._current_task_id = 0
        yield from super().query_embed(query, **kwargs)

    def passage_embed(self, texts: Iterable[str], **kwargs) -> Iterable[np.ndarray]:
        self._current_task_id = 1
        yield from super().passage_embed(texts, **kwargs)


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
