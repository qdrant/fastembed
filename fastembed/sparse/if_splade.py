import json
from typing import Any, Iterable, Sequence, Type

import numpy as np

from fastembed.common import OnnxProvider
from fastembed.common.model_description import ModelSource, SparseModelDescription
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.preprocessor_utils import load_tokenizer
from fastembed.common.types import Device
from fastembed.common.utils import define_cache_dir, iter_batch
from fastembed.sparse.sparse_embedding_base import (
    SparseEmbedding,
    SparseTextEmbeddingBase,
)
from fastembed.text.onnx_text_model import OnnxTextModel, TextEmbeddingWorker

IDF_FILE = "idf.json"

supported_if_splade_models: list[SparseModelDescription] = [
    SparseModelDescription(
        model="opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte",
        vocab_size=30522,
        description="Inference-free SPLADE model. Documents are expanded with an ONNX encoder at index "
        "time, queries are encoded with a tokenizer and an IDF lookup table only, "
        "without any model inference.",
        license="apache-2.0",
        size_in_GB=0.55,
        sources=ModelSource(hf="Qdrant/opensearch-neural-sparse-encoding-doc-v3-gte"),
        model_file="model.onnx",
        additional_files=[IDF_FILE],
        requires_idf=None,
    ),
]


class IfSplade(SparseTextEmbeddingBase, OnnxTextModel[SparseEmbedding]):
    """Inference-free (asymmetric) SPLADE model.

    Documents are encoded with a neural encoder which expands them into a sparse vocabulary-sized
    vector, while queries are encoded by tokenizing the text and looking up a precomputed IDF
    weight per token — no neural inference happens at query time.

    Query and document embeddings are compared with a dot product.
    Special tokens are excluded from both document and query embeddings.
    """

    def _post_process_onnx_output(
        self, output: OnnxOutputContext, **kwargs: Any
    ) -> Iterable[SparseEmbedding]:
        if output.attention_mask is None:
            raise ValueError("attention_mask must be provided for document post-processing")

        # Max-pool token logits over the sequence, masking out the padding
        pooled = np.max(
            output.model_output * np.expand_dims(output.attention_mask, axis=-1), axis=1
        )
        # v3 models of the opensearch-neural-sparse family use a double log activation,
        # log(1 + log(1 + relu(x))), to increase sparsity of document embeddings
        scores = np.log1p(np.log1p(np.maximum(pooled, 0.0)))

        if self.special_tokens_ids:
            scores[:, list(self.special_tokens_ids)] = 0.0

        for row_scores in scores:
            indices = row_scores.nonzero()[0]
            yield SparseEmbedding(values=row_scores[indices], indices=indices)

    def token_count(
        self, texts: str | Iterable[str], batch_size: int = 1024, **kwargs: Any
    ) -> int:
        # unlike `OnnxTextModel._token_count`, does not require the onnx model to be loaded
        token_num = 0
        texts = [texts] if isinstance(texts, str) else texts
        for batch in iter_batch(texts, batch_size):
            for tokens in self.tokenizer.encode_batch(batch):  # type: ignore[union-attr]
                token_num += sum(tokens.attention_mask)
        return token_num

    @classmethod
    def _list_supported_models(cls) -> list[SparseModelDescription]:
        """Lists the supported models.

        Returns:
            list[SparseModelDescription]: A list of SparseModelDescription objects containing the model information.
        """
        return supported_if_splade_models

    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        threads: int | None = None,
        providers: Sequence[OnnxProvider] | None = None,
        cuda: bool | Device = Device.AUTO,
        device_ids: list[int] | None = None,
        lazy_load: bool = False,
        device_id: int | None = None,
        specific_model_path: str | None = None,
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
            cuda (Union[bool, Device], optional): Whether to use cuda for inference. Mutually exclusive with `providers`
                Defaults to Device.
            device_ids (Optional[list[int]], optional): The list of device ids to use for data parallel processing in
                workers. Should be used with `cuda` equals to `True`, `Device.AUTO` or `Device.CUDA`, mutually exclusive
                with `providers`. Defaults to None.
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
        self._extra_session_options = self._select_exposed_session_options(kwargs)

        # List of device ids, that can be used for data parallel processing in workers
        self.device_ids = device_ids
        self.cuda = cuda

        # This device_id will be used if we need to load model in current process
        self.device_id: int | None = None
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

        # The tokenizer and the idf table are lightweight and are required for query embedding,
        # which does not involve any model inference, so they are loaded eagerly, while
        # `lazy_load` only defers the initialization of the onnx model
        self.tokenizer, self.special_token_to_id = load_tokenizer(model_dir=self._model_dir)
        self.special_tokens_ids: set[int] = set(self.special_token_to_id.values())
        self._token_id_to_idf = self._load_idf()

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
            extra_session_options=self._extra_session_options,
        )

    def _load_idf(self) -> dict[int, float]:
        with open(self._model_dir / IDF_FILE) as f:
            token_to_idf: dict[str, float] = json.load(f)

        vocab: dict[str, int] = self.tokenizer.get_vocab()  # type: ignore[union-attr]
        return {vocab[token]: idf for token, idf in token_to_idf.items() if token in vocab}

    def embed(
        self,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[SparseEmbedding]:
        """
        Encode a list of documents into list of embeddings.

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
            extra_session_options=self._extra_session_options,
            **kwargs,
        )

    def query_embed(self, query: str | Iterable[str], **kwargs: Any) -> Iterable[SparseEmbedding]:
        """
        Encode a list of queries into list of sparse embeddings without any model inference.

        A query is tokenized, and each unique token is assigned its IDF weight from
        a precomputed lookup table shipped with the model. Special tokens are ignored.
        """
        if isinstance(query, str):
            query = [query]

        for text in query:
            token_ids = set(self.tokenizer.encode(text).ids) - self.special_tokens_ids  # type: ignore[union-attr]
            embedding = {
                token_id: self._token_id_to_idf[token_id]
                for token_id in sorted(token_ids)
                if token_id in self._token_id_to_idf
            }
            yield SparseEmbedding.from_dict(embedding)

    @classmethod
    def _get_worker_class(cls) -> Type[TextEmbeddingWorker[SparseEmbedding]]:
        return IfSpladeEmbeddingWorker


class IfSpladeEmbeddingWorker(TextEmbeddingWorker[SparseEmbedding]):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs: Any) -> IfSplade:
        return IfSplade(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
