import string
from typing import Any, Iterable, Optional, Sequence, Type, Union

import numpy as np
from tokenizers import Encoding, Tokenizer

from fastembed.common.preprocessor_utils import load_tokenizer
from fastembed.common.types import NumpyArray
from fastembed.common import OnnxProvider
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.utils import define_cache_dir
from fastembed.late_interaction.late_interaction_embedding_base import (
    LateInteractionTextEmbeddingBase,
)
from fastembed.text.onnx_text_model import OnnxTextModel, TextEmbeddingWorker
from fastembed.common.model_description import DenseModelDescription, ModelSource

supported_colbert_models: list[DenseModelDescription] = [
    DenseModelDescription(
        model="colbert-ir/colbertv2.0",
        dim=128,
        description="Late interaction model",
        license="mit",
        size_in_GB=0.44,
        sources=ModelSource(hf="colbert-ir/colbertv2.0"),
        model_file="model.onnx",
    ),
    DenseModelDescription(
        model="answerdotai/answerai-colbert-small-v1",
        dim=96,
        description="Text embeddings, Unimodal (text), Multilingual (~100 languages), 512 input tokens truncation, 2024 year",
        license="apache-2.0",
        size_in_GB=0.13,
        sources=ModelSource(hf="answerdotai/answerai-colbert-small-v1"),
        model_file="vespa_colbert.onnx",
    ),
]


class Colbert(LateInteractionTextEmbeddingBase, OnnxTextModel[NumpyArray]):
    QUERY_MARKER_TOKEN_ID = 1
    DOCUMENT_MARKER_TOKEN_ID = 2
    MIN_QUERY_LENGTH = 31  # it's 32, we add one additional special token in the beginning
    MASK_TOKEN = "[MASK]"

    def _post_process_onnx_output(
        self, output: OnnxOutputContext, is_doc: bool = True, **kwargs: Any
    ) -> Iterable[NumpyArray]:
        if not is_doc:
            for embedding in output.model_output:
                yield embedding
        else:
            if output.input_ids is None or output.attention_mask is None:
                raise ValueError(
                    "input_ids and attention_mask must be provided for document post-processing"
                )

            for i, token_sequence in enumerate(output.input_ids):
                for j, token_id in enumerate(token_sequence):  # type: ignore
                    if token_id in self.skip_list or token_id == self.pad_token_id:
                        output.attention_mask[i, j] = 0

            output.model_output *= np.expand_dims(output.attention_mask, 2)
            norm = np.linalg.norm(output.model_output, ord=2, axis=2, keepdims=True)
            norm_clamped = np.maximum(norm, 1e-12)
            output.model_output /= norm_clamped

            for embedding, attention_mask in zip(output.model_output, output.attention_mask):
                yield embedding[attention_mask == 1]

    def _preprocess_onnx_input(
        self, onnx_input: dict[str, NumpyArray], is_doc: bool = True, **kwargs: Any
    ) -> dict[str, NumpyArray]:
        marker_token = self.DOCUMENT_MARKER_TOKEN_ID if is_doc else self.QUERY_MARKER_TOKEN_ID
        onnx_input["input_ids"] = np.insert(
            onnx_input["input_ids"].astype(np.int64), 1, marker_token, axis=1
        )
        onnx_input["attention_mask"] = np.insert(
            onnx_input["attention_mask"].astype(np.int64), 1, 1, axis=1
        )
        return onnx_input

    def tokenize(self, documents: list[str], is_doc: bool = True, **kwargs: Any) -> list[Encoding]:
        return (
            self._tokenize_documents(documents=documents)
            if is_doc
            else self._tokenize_query(query=next(iter(documents)))
        )

    def _tokenize_query(self, query: str) -> list[Encoding]:
        assert self.query_tokenizer is not None
        encoded = self.query_tokenizer.encode_batch([query])
        return encoded

    def _tokenize_documents(self, documents: list[str]) -> list[Encoding]:
        encoded = self.tokenizer.encode_batch(documents)  # type: ignore[union-attr]
        return encoded

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        """Lists the supported models.

        Returns:
            list[DenseModelDescription]: A list of DenseModelDescription objects containing the model information.
        """
        return supported_colbert_models

    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence[OnnxProvider]] = None,
        cuda: bool = False,
        device_ids: Optional[list[int]] = None,
        lazy_load: bool = False,
        device_id: Optional[int] = None,
        specific_model_path: Optional[str] = None,
        **kwargs: Any,
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
            device_ids (Optional[list[int]], optional): The list of device ids to use for data parallel processing in
                workers. Should be used with `cuda=True`, mutually exclusive with `providers`. Defaults to None.
            lazy_load (bool, optional): Whether to load the model during class initialization or on demand.
                Should be set to True when using multiple-gpu and parallel encoding. Defaults to False.
            device_id (Optional[int], optional): The device id to use for loading the model in the worker process.
            specific_model_path (Optional[str], optional): The specific path to the onnx model dir if it should be imported from somewhere else

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
        self.device_id: Optional[int] = None
        if device_id is not None:
            self.device_id = device_id
        elif self.device_ids is not None:
            self.device_id = self.device_ids[0]

        self.model_description = self._get_model_description(model_name)
        self.cache_dir = str(define_cache_dir(cache_dir))

        self._specific_model_path = specific_model_path
        self._model_dir = self.download_model(
            self.model_description,
            self.cache_dir,
            local_files_only=self._local_files_only,
            specific_model_path=self._specific_model_path,
        )
        self.mask_token_id: Optional[int] = None
        self.pad_token_id: Optional[int] = None
        self.skip_list: set[int] = set()

        self.query_tokenizer: Optional[Tokenizer] = None

        if not self.lazy_load:
            self.load_onnx_model()

    def load_onnx_model(self) -> None:
        self._load_onnx_model(
            model_dir=self._model_dir,
            model_file=self.model_description.model_file,
            threads=self.threads,
            providers=self.providers,
            cuda=self.cuda,
            device_id=self.device_id,
        )
        self.query_tokenizer, _ = load_tokenizer(model_dir=self._model_dir)

        assert self.tokenizer is not None
        self.mask_token_id = self.special_token_to_id[self.MASK_TOKEN]
        self.pad_token_id = self.tokenizer.padding["pad_id"]
        self.skip_list = {
            self.tokenizer.encode(symbol, add_special_tokens=False).ids[0]
            for symbol in string.punctuation
        }
        current_max_length = self.tokenizer.truncation["max_length"]
        # ensure not to overflow after adding document-marker
        self.tokenizer.enable_truncation(max_length=current_max_length - 1)
        self.query_tokenizer.enable_truncation(max_length=current_max_length - 1)
        self.query_tokenizer.enable_padding(
            pad_token=self.MASK_TOKEN,
            pad_id=self.mask_token_id,
            length=self.MIN_QUERY_LENGTH,
        )

    def embed(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
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
            local_files_only=self._local_files_only,
            specific_model_path=self._specific_model_path,
            **kwargs,
        )

    def query_embed(self, query: Union[str, Iterable[str]], **kwargs: Any) -> Iterable[NumpyArray]:
        if isinstance(query, str):
            query = [query]

        if not hasattr(self, "model") or self.model is None:
            self.load_onnx_model()

        for text in query:
            yield from self._post_process_onnx_output(
                self.onnx_embed([text], is_doc=False), is_doc=False
            )

    @classmethod
    def _get_worker_class(cls) -> Type[TextEmbeddingWorker[NumpyArray]]:
        return ColbertEmbeddingWorker


class ColbertEmbeddingWorker(TextEmbeddingWorker[NumpyArray]):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs: Any) -> Colbert:
        return Colbert(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
