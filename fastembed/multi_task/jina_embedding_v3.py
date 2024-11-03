import os
from multiprocessing import get_all_start_methods
from typing import Any, Dict, Iterable, List, Optional, Sequence, Type, Union, Tuple

import numpy as np

from fastembed.common import OnnxProvider
from fastembed.common.utils import iter_batch
from fastembed.common.onnx_model import OnnxOutputContext, T
from fastembed.text.onnx_text_model import OnnxTextModel, TextEmbeddingWorker
from fastembed.multi_task.multi_task_embedding_base import MultiTaskTextEmbeddingBase

from fastembed.parallel_processor import ParallelWorkerPool

supported_jina_embedding_models = [
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
    }
]


class JinaEmbeddingV3(MultiTaskTextEmbeddingBase, OnnxTextModel[np.ndarray]):
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

    def get_task_type_dict(self) -> Dict[str, int]:
        """Returns the available task types

        Returns:
            Dict[str, int]: A dictionary containing the task types and their corresponding indices."""
        return self.model_description["tasks"]

    def _post_process_onnx_output(self, output: OnnxOutputContext) -> Iterable[np.ndarray]:
        if output.attention_mask is None:
            raise ValueError("attention_mask must be provided for output processing")

        embeddings = output.model_output
        attn_mask = output.attention_mask
        return self.mean_pooling(embeddings, attn_mask).astype(np.float32)

    def onnx_embed(
        self,
        documents: List[str],
        task_id: int,
        **kwargs,
    ) -> OnnxOutputContext:
        encoded = self.tokenize(documents, **kwargs)
        input_ids = np.array([e.ids for e in encoded])
        attention_mask = np.array([e.attention_mask for e in encoded])
        input_names = {node.name for node in self.model.get_inputs()}

        assert "task_id" in input_names, "task_id must be provided for input"

        onnx_input = {
            "input_ids": np.array(input_ids, dtype=np.int64),
            "task_id": np.array(task_id, dtype=np.int64),
        }
        if "attention_mask" in input_names:
            onnx_input["attention_mask"] = np.array(attention_mask, dtype=np.int64)
        if "token_type_ids" in input_names:
            onnx_input["token_type_ids"] = np.array(
                [np.zeros(len(e), dtype=np.int64) for e in input_ids], dtype=np.int64
            )

        onnx_input = self._preprocess_onnx_input(onnx_input, **kwargs)

        model_output = self.model.run(self.ONNX_OUTPUT_NAMES, onnx_input)
        return OnnxOutputContext(
            model_output=model_output[0],
            attention_mask=onnx_input.get("attention_mask", attention_mask),
            input_ids=onnx_input.get("input_ids", input_ids),
        )

    def _embed_documents(
        self,
        model_name: str,
        cache_dir: str,
        documents: Union[str, Iterable[str]],
        task_id: int,
        batch_size: int = 256,
        parallel: Optional[int] = None,
        providers: Optional[Sequence[OnnxProvider]] = None,
        cuda: bool = False,
        device_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> Iterable[T]:
        is_small = False

        if isinstance(documents, str):
            documents = [documents]
            is_small = True

        if isinstance(documents, list):
            if len(documents) < batch_size:
                is_small = True

        if parallel is None or is_small:
            if not hasattr(self, "model") or self.model is None:
                self.load_onnx_model()
            for batch in iter_batch(documents, batch_size):
                yield from self._post_process_onnx_output(self.onnx_embed(batch, task_id))
        else:
            if parallel == 0:
                parallel = os.cpu_count()

            start_method = "forkserver" if "forkserver" in get_all_start_methods() else "spawn"
            params = {
                "model_name": model_name,
                "cache_dir": cache_dir,
                "providers": providers,
                **kwargs,
            }

            pool = ParallelWorkerPool(
                num_workers=parallel or 1,
                worker=self._get_worker_class(),
                cuda=cuda,
                device_ids=device_ids,
                start_method=start_method,
            )
            batches_with_task = ((batch, task_id) for batch in iter_batch(documents, batch_size))
            for batch in pool.ordered_map(batches_with_task, **params):
                yield from self._post_process_onnx_output(batch)

    def task_embed(
        self,
        documents: Union[str, Iterable[str]],
        task_type: str,
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs,
    ) -> Iterable[np.ndarray]:
        if task_type not in self.model_description["tasks"]:
            raise ValueError(
                f"Unsupported task type: {task_type}. "
                f"Supported types: {list(self.model_description['tasks'].keys())}"
            )

        task_id = self.get_task_type_dict()[task_type]

        if isinstance(documents, str):
            documents = [documents]

        yield from self._embed_documents(
            model_name=self.model_name,
            cache_dir=str(self.cache_dir),
            documents=documents,
            task_id=task_id,
            batch_size=batch_size,
            parallel=parallel,
            providers=self.providers,
            cuda=self.cuda,
            device_ids=self.device_ids,
            **kwargs,
        )


class JinaEmbeddingV3Worker(TextEmbeddingWorker):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs) -> JinaEmbeddingV3:
        return JinaEmbeddingV3(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )

    def process(
        self, items: Iterable[Tuple[int, Tuple[List[str], int]]]
    ) -> Iterable[Tuple[int, OnnxOutputContext]]:
        for idx, (batch, task_id) in items:
            onnx_output = self.model.onnx_embed(batch, task_id)
            yield idx, onnx_output
