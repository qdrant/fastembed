import os
from multiprocessing import get_all_start_methods
from typing import Any, Dict, Iterable, List, Optional, Sequence, Type, Union, Tuple

import numpy as np

from fastembed.common import OnnxProvider
from fastembed.common.utils import iter_batch, define_cache_dir
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.text.onnx_text_model import OnnxTextModel, TextEmbeddingWorker
from fastembed.multi_task.multi_task_embedding_base import (
    MultiTaskTextEmbeddingBase,
    MultiTaskEmbedding,
)

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
        device_id: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            model_name (str): The name of the model to use.
            cache_dir (str, optional): The path to the cache directory.
                                       Can be set using the `FASTEMBED_CACHE_PATH` env variable.
                                       Defaults to `fastembed_cache` in the system's temp directory.
            threads (int, optional): The number of threads single onnxruntime session can use. Defaults to None.
            providers (Optional[Sequence[OnnxProvider]], optional): The list of onnxruntime providers to use.
                Mutually exclusive with the `cuda` and `device_ids` arguments. Defaults to None.
            cuda (bool, optional): Whether to use cuda for inference. Mutually exclusive with `providers`
                Defaults to False.
            device_ids (Optional[List[int]], optional): The list of device ids to use for data parallel processing in
                workers. Should be used with `cuda=True`, mutually exclusive with `providers`. Defaults to None.
            lazy_load (bool, optional): Whether to load the model during class initialization or on demand.
                Should be set to True when using multiple-gpu and parallel encoding. Defaults to False.
            device_id (Optional[int], optional): The device id to use for loading the model in the worker process.

        Raises:
            ValueError: If the model_name is not in the format <org>/<model> e.g. BAAI/bge-base-en.
        """
        super().__init__(model_name, cache_dir, threads, **kwargs)
        self.providers = providers
        self.lazy_load = lazy_load

        # List of device ids, that can be used for data parallel processing in workers
        self.device_ids = device_ids
        self.cuda = cuda

        # This device_id will be used if we need to load model in current process
        if device_id is not None:
            self.device_id = device_id
        elif self.device_ids is not None:
            self.device_id = self.device_ids[0]
        else:
            self.device_id = None

        self.model_description = self._get_model_description(model_name)
        self.cache_dir = define_cache_dir(cache_dir)
        self._model_dir = self.download_model(
            self.model_description, self.cache_dir, local_files_only=self._local_files_only
        )

        if not self.lazy_load:
            self.load_onnx_model()

    def load_onnx_model(self) -> None:
        self._load_onnx_model(
            model_dir=self._model_dir,
            model_file=self.model_description["model_file"],
            threads=self.threads,
            providers=self.providers,
            cuda=self.cuda,
            device_id=self.device_id,
        )

    def get_task_types_dict(self) -> Dict[str, int]:
        """Returns the available task types

        Returns:
            Dict[str, int]: A dictionary containing the task types and their corresponding indices."""
        return self.model_description["tasks"]

    def get_task_type_from_id(self, task_id: int) -> str:
        """Get task type string from task ID

        Args:
            task_id (int): The task ID
        Returns:
            str: The task type string.
        """
        for task_type, tid in self.model_description["tasks"].items():
            if tid == task_id:
                return task_type
        raise ValueError(f"Unknown task ID: {task_id}")

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
    ) -> Iterable[MultiTaskEmbedding]:
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
                embeddings = self._post_process_onnx_output(self.onnx_embed(batch, task_id))
                for embedding in embeddings:
                    yield MultiTaskEmbedding(
                        embedding=embedding,
                        task_type=self.get_task_type_from_id(task_id),
                        dimension=embedding.shape[-1],
                    )
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
                embeddings = self._post_process_onnx_output(batch)
                for embedding in embeddings:
                    yield MultiTaskEmbedding(
                        embedding=embedding,
                        task_type=self.get_task_type_from_id(task_id),
                        dimension=embedding.shape[-1],
                    )

    def task_embed(
        self,
        documents: Union[str, Iterable[str]],
        task_type: str,
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs,
    ) -> Iterable[MultiTaskEmbedding]:
        if task_type not in self.model_description["tasks"]:
            raise ValueError(
                f"Unsupported task type: {task_type}. Supported types: {list(self.model_description['tasks'].keys())}"
            )

        task_id = self.get_task_types_dict()[task_type]

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
