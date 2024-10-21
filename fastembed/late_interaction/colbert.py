import string
from typing import Any, Dict, Iterable, List, Optional, Sequence, Type, Union

import numpy as np
from tokenizers import Encoding

from fastembed.common import OnnxProvider
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.utils import define_cache_dir
from fastembed.late_interaction.late_interaction_embedding_base import (
    LateInteractionTextEmbeddingBase,
)
from fastembed.text.onnx_text_model import OnnxTextModel, TextEmbeddingWorker

supported_colbert_models = [
    {
        "model": "colbert-ir/colbertv2.0",
        "dim": 128,
        "description": "Late interaction model",
        "license": "mit",
        "size_in_GB": 0.44,
        "sources": {
            "hf": "colbert-ir/colbertv2.0",
        },
        "model_file": "model.onnx",
    },
    {
        "model": "answerdotai/answerai-colbert-small-v1",
        "dim": 96,
        "description": "Text embeddings, Unimodal (text), Multilingual (~100 languages), 512 input tokens truncation, 2024 year",
        "license": "apache-2.0",
        "size_in_GB": 0.13,
        "sources": {
            "hf": "answerdotai/answerai-colbert-small-v1",
        },
        "model_file": "vespa_colbert.onnx",
    },
]


class Colbert(LateInteractionTextEmbeddingBase, OnnxTextModel[np.ndarray]):
    QUERY_MARKER_TOKEN_ID = 1
    DOCUMENT_MARKER_TOKEN_ID = 2
    MIN_QUERY_LENGTH = 32
    MASK_TOKEN = "[MASK]"

    def _post_process_onnx_output(
        self, output: OnnxOutputContext, is_doc: bool = True
    ) -> Iterable[np.ndarray]:
        if not is_doc:
            return output.model_output.astype(np.float32)

        if output.input_ids is None or output.attention_mask is None:
            raise ValueError(
                "input_ids and attention_mask must be provided for document post-processing"
            )

        for i, token_sequence in enumerate(output.input_ids):
            for j, token_id in enumerate(token_sequence):
                if token_id in self.skip_list or token_id == self.pad_token_id:
                    output.attention_mask[i, j] = 0

        output.model_output *= np.expand_dims(output.attention_mask, 2).astype(np.float32)
        norm = np.linalg.norm(output.model_output, ord=2, axis=2, keepdims=True)
        norm_clamped = np.maximum(norm, 1e-12)
        output.model_output /= norm_clamped
        return output.model_output.astype(np.float32)

    def _preprocess_onnx_input(
        self, onnx_input: Dict[str, np.ndarray], is_doc: bool = True
    ) -> Dict[str, np.ndarray]:
        if is_doc:
            onnx_input["input_ids"][:, 1] = self.DOCUMENT_MARKER_TOKEN_ID
        else:
            onnx_input["input_ids"][:, 1] = self.QUERY_MARKER_TOKEN_ID
        return onnx_input

    def tokenize(self, documents: List[str], is_doc: bool = True) -> List[Encoding]:
        return (
            self._tokenize_documents(documents=documents)
            if is_doc
            else self._tokenize_query(query=next(iter(documents)))
        )

    def _tokenize_query(self, query: str) -> List[Encoding]:
        # ". " is added to a query to be replaced with a special query token
        query = [f". {query}"]
        encoded = self.tokenizer.encode_batch(query)
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
            encoded = self.tokenizer.encode_batch(query)
            if prev_padding is None:
                self.tokenizer.no_padding()
            else:
                self.tokenizer.enable_padding(**prev_padding)
        return encoded

    def _tokenize_documents(self, documents: List[str]) -> List[Encoding]:
        # ". " is added to a document to be replaced with a special document token
        documents = [". " + doc for doc in documents]
        encoded = self.tokenizer.encode_batch(documents)
        return encoded

    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        """Lists the supported models.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the model information.
        """
        return supported_colbert_models

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
        self.mask_token_id = None
        self.pad_token_id = None
        self.skip_list = set()

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
        self.mask_token_id = self.special_token_to_id["[MASK]"]
        self.pad_token_id = self.tokenizer.padding["pad_id"]
        self.skip_list = {
            self.tokenizer.encode(symbol, add_special_tokens=False).ids[0]
            for symbol in string.punctuation
        }

    def embed(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs,
    ) -> Iterable[np.ndarray]:
        """
        Encode a list of documents into list of embeddings.
        We use mean pooling with attention so that the model can handle variable-length inputs.

        Args:
            documents: Iterator of documents or single document to embed
            batch_size: Batch size for encoding -- higher values will use more memory, but be faster
            parallel:
                If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
                If 0, use all available cores.
                If None, don't use data-parallel processing, use default onnxruntime threading instead.

        Returns:
            List of embeddings, one per document
        """
        yield from self._embed_documents(
            model_name=self.model_name,
            cache_dir=str(self.cache_dir),
            documents=documents,
            batch_size=batch_size,
            parallel=parallel,
            providers=self.providers,
            cuda=self.cuda,
            device_ids=self.device_ids,
            **kwargs,
        )

    def query_embed(self, query: Union[str, List[str]], **kwargs) -> Iterable[np.ndarray]:
        if isinstance(query, str):
            query = [query]

        if not hasattr(self, "model") or self.model is None:
            self.load_onnx_model()

        for text in query:
            yield from self._post_process_onnx_output(
                self.onnx_embed([text], is_doc=False), is_doc=False
            )

    @classmethod
    def _get_worker_class(cls) -> Type[TextEmbeddingWorker]:
        return ColbertEmbeddingWorker


class ColbertEmbeddingWorker(TextEmbeddingWorker):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs) -> Colbert:
        return Colbert(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
