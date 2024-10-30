from typing import Any, Dict, Iterable, List, Optional, Sequence, Type, Union

import numpy as np

from fastembed.common import OnnxProvider
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.text.onnx_text_model import OnnxTextModel, TextEmbeddingWorker
from fastembed.multi_task.multi_task_embedding_base import MultiTaskTextEmbeddingBase

supported_jina_embedding_models = [
    {
        "model": "jinaai/jina-embeddings-v3",
        "dim": [32, 64, 128, 256, 512, 768, 1024],
        "description": "Multi-task, multi-lingual embedding model with Matryoshka architecture",
        "license": "cc-by-nc-4.0",
        "size_in_GB": 2.29,
        "sources": {
            "hf": "jinaai/jina-embeddings-v3",
        },
        "model_file": "onnx/model.onnx",
        "additional_files": ["onnx/model.onnx_data"],
    }
]


class JinaEmbeddingV3(MultiTaskTextEmbeddingBase, OnnxTextModel[np.ndarray]):
    TASK_TYPES = [
        "retrieval.query",
        "retrieval.passage",
        "separation",
        "classification",
        "text-matching",
    ]

    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        """Lists the supported models.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the model information.
        """
        return supported_jina_embedding_models

    @classmethod
    def _get_worker_class(cls) -> Type[TextEmbeddingWorker]:
        return JinaEmbeddingV3Worker

    @classmethod
    def mean_pooling(cls, model_output: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        token_embeddings = model_output
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
        input_mask_expanded = np.tile(input_mask_expanded, (1, 1, token_embeddings.shape[-1]))
        input_mask_expanded = input_mask_expanded.astype(float)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.sum(input_mask_expanded, axis=1)
        pooled_embeddings = sum_embeddings / np.maximum(sum_mask, 1e-9)
        return pooled_embeddings

    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence[OnnxProvider]] = None,
        cuda: bool = False,
        device_ids: Optional[List[int]] = None,
        lazy_load: bool = False,
        **kwargs,
    ):
        super().__init__(model_name, cache_dir, threads, **kwargs)
        self.providers = providers
        self.lazy_load = lazy_load
        self.device_ids = device_ids
        self.cuda = cuda
        self.device_id = device_ids[0] if device_ids else None

        self.model_description = self._get_model_description(model_name)
        if not self.lazy_load:
            self.load_onnx_model()

    def _post_process_onnx_output(self, output: OnnxOutputContext) -> Iterable[np.ndarray]:
        if output.attention_mask is None:
            raise ValueError("attention_mask must be provided for output processing")

        embeddings = output.model_output
        attn_mask = output.attention_mask
        return self.mean_pooling(embeddings, attn_mask).astype(np.float32)

    def task_embed(
        self,
        documents: Union[str, Iterable[str]],
        task_type: str,
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs,
    ) -> Iterable[np.ndarray]:
        if task_type not in self.TASK_TYPES:
            raise ValueError(f"Invalid task type: {task_type}")

        if isinstance(documents, str):
            documents = [documents]

        yield "dummy"


class JinaEmbeddingV3Worker(TextEmbeddingWorker):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs) -> JinaEmbeddingV3:
        return JinaEmbeddingV3(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
