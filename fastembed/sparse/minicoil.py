from pathlib import Path

from typing import Any, Optional, Sequence, Iterable, Union, Type

import numpy as np
from numpy.typing import NDArray
from py_rust_stemmers import SnowballStemmer
from tokenizers import Tokenizer

from fastembed.common.model_description import SparseModelDescription, ModelSource
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common import OnnxProvider
from fastembed.common.utils import define_cache_dir
from fastembed.sparse.sparse_embedding_base import (
    SparseEmbedding,
    SparseTextEmbeddingBase,
)
from fastembed.sparse.utils.minicoil_encoder import Encoder
from fastembed.sparse.utils.sparse_vectors_converter import SparseVectorConverter, WordEmbedding
from fastembed.sparse.utils.vocab_resolver import VocabResolver, VocabTokenizer
from fastembed.text.onnx_text_model import OnnxTextModel, TextEmbeddingWorker


MINICOIL_MODEL_FILE = "minicoil.triplet.model.npy"
MINICOIL_VOCAB_FILE = "minicoil.triplet.model.vocab"
STOPWORDS_FILE = "stopwords.txt"


supported_minicoil_models: list[SparseModelDescription] = [
    SparseModelDescription(
        model="Qdrant/minicoil-v1",
        vocab_size=19125,
        description="Sparse embedding model, that resolves semantic meaning of the words, "
        "while keeping exact keyword match behavior. "
        "Based on jinaai/jina-embeddings-v2-small-en-tokens",
        license="apache-2.0",
        size_in_GB=0.09,
        sources=ModelSource(hf="Qdrant/minicoil-v1"),
        model_file="onnx/model.onnx",
        additional_files=[
            STOPWORDS_FILE,
            MINICOIL_MODEL_FILE,
            MINICOIL_VOCAB_FILE,
        ],
        requires_idf=True,
    ),
]

MODEL_TO_LANGUAGE = {
    "Qdrant/minicoil-v1": "english",
}


class MiniCOIL(SparseTextEmbeddingBase, OnnxTextModel[SparseEmbedding]):
    """
        MiniCOIL is a sparse embedding model, that resolves semantic meaning of the words,
        while keeping exact keyword match behavior.

        Each vocabulary token is converted into 4d component of a sparse vector, which is then weighted by the token frequency in the corpus.
        If the token is not found in the corpus, it is treated exactly like in BM25.
    `
        The model is based on `jinaai/jina-embeddings-v2-small-en-tokens`
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence[OnnxProvider]] = None,
        k: float = 1.2,
        b: float = 0.75,
        avg_len: float = 150.0,
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
            providers (Optional[Sequence[OnnxProvider]], optional): The providers to use for onnxruntime.
            k (float, optional): The k parameter in the BM25 formula. Defines the saturation of the term frequency.
                I.e. defines how fast the moment when additional terms stop to increase the score. Defaults to 1.2.
            b (float, optional): The b parameter in the BM25 formula. Defines the importance of the document length.
                Defaults to 0.75.
            avg_len (float, optional): The average length of the documents in the corpus. Defaults to 150.0.
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
        self.device_ids = device_ids
        self.cuda = cuda
        self.device_id = device_id
        self.k = k
        self.b = b
        self.avg_len = avg_len

        # Initialize class attributes
        self.tokenizer: Optional[Tokenizer] = None
        self.invert_vocab: dict[int, str] = {}
        self.special_tokens: set[str] = set()
        self.special_tokens_ids: set[int] = set()
        self.stopwords: set[str] = set()
        self.vocab_resolver: Optional[VocabResolver] = None
        self.encoder: Optional[Encoder] = None
        self.output_dim: Optional[int] = None
        self.sparse_vector_converter: Optional[SparseVectorConverter] = None

        self.model_description = self._get_model_description(model_name)
        self.cache_dir = str(define_cache_dir(cache_dir))
        self._specific_model_path = specific_model_path
        self._model_dir = self.download_model(
            self.model_description,
            self.cache_dir,
            local_files_only=self._local_files_only,
            specific_model_path=self._specific_model_path,
        )

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

        assert self.tokenizer is not None

        for token, idx in self.tokenizer.get_vocab().items():  # type: ignore[union-attr]
            self.invert_vocab[idx] = token
        self.special_tokens = set(self.special_token_to_id.keys())
        self.special_tokens_ids = set(self.special_token_to_id.values())
        self.stopwords = set(self._load_stopwords(self._model_dir))

        stemmer = SnowballStemmer(MODEL_TO_LANGUAGE[self.model_name])

        self.vocab_resolver = VocabResolver(
            tokenizer=VocabTokenizer(self.tokenizer),
            stopwords=self.stopwords,
            stemmer=stemmer,
        )
        self.vocab_resolver.load_json_vocab(str(self._model_dir / MINICOIL_VOCAB_FILE))

        weights = np.load(str(self._model_dir / MINICOIL_MODEL_FILE), mmap_mode="r")
        self.encoder = Encoder(weights)
        self.output_dim = self.encoder.output_dim

        self.sparse_vector_converter = SparseVectorConverter(
            stopwords=self.stopwords,
            stemmer=stemmer,
            k=self.k,
            b=self.b,
            avg_len=self.avg_len,
        )

    def embed(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterable[SparseEmbedding]:
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
            k=self.k,
            b=self.b,
            avg_len=self.avg_len,
            is_query=False,
            local_files_only=self._local_files_only,
            specific_model_path=self._specific_model_path,
            **kwargs,
        )

    def query_embed(
        self, query: Union[str, Iterable[str]], **kwargs: Any
    ) -> Iterable[SparseEmbedding]:
        """
        Encode a list of queries into list of embeddings.
        """
        yield from self._embed_documents(
            model_name=self.model_name,
            cache_dir=str(self.cache_dir),
            documents=query,
            providers=self.providers,
            cuda=self.cuda,
            device_ids=self.device_ids,
            k=self.k,
            b=self.b,
            avg_len=self.avg_len,
            is_query=True,
            local_files_only=self._local_files_only,
            specific_model_path=self._specific_model_path,
            **kwargs,
        )

    @classmethod
    def _load_stopwords(cls, model_dir: Path) -> list[str]:
        stopwords_path = model_dir / STOPWORDS_FILE
        if not stopwords_path.exists():
            return []

        with open(stopwords_path, "r") as f:
            return f.read().splitlines()

    @classmethod
    def _list_supported_models(cls) -> list[SparseModelDescription]:
        """Lists the supported models.

        Returns:
            list[SparseModelDescription]: A list of SparseModelDescription objects containing the model information.
        """
        return supported_minicoil_models

    def _post_process_onnx_output(
        self, output: OnnxOutputContext, is_query: bool = False, **kwargs: Any
    ) -> Iterable[SparseEmbedding]:
        if output.input_ids is None:
            raise ValueError("input_ids must be provided for document post-processing")

        assert self.vocab_resolver is not None
        assert self.encoder is not None
        assert self.sparse_vector_converter is not None

        # Size: (batch_size, sequence_length, hidden_size)
        embeddings = output.model_output
        # Size: (batch_size, sequence_length)
        assert output.attention_mask is not None
        masks = output.attention_mask

        vocab_size = self.vocab_resolver.vocab_size()
        embedding_size = self.encoder.output_dim

        # For each document we only select those embeddings that are not masked out

        for i in range(embeddings.shape[0]):
            # Size: (sequence_length, hidden_size)
            token_embeddings = embeddings[i, masks[i] == 1]

            # Size: (sequence_length)
            token_ids: NDArray[np.int64] = output.input_ids[i, masks[i] == 1]

            word_ids_array, counts, oov, forms = self.vocab_resolver.resolve_tokens(token_ids)

            # Size: (1, words)
            word_ids_array_expanded: NDArray[np.int64] = np.expand_dims(word_ids_array, axis=0)

            # Size: (1, words, embedding_size)
            token_embeddings_array: NDArray[np.float32] = np.expand_dims(token_embeddings, axis=0)

            assert word_ids_array_expanded.shape[1] == token_embeddings_array.shape[1]

            # Size of word_ids_mapping: (unique_words, 2) - [vocab_id, batch_id]
            # Size of embeddings: (unique_words, embedding_size)
            ids_mapping, minicoil_embeddings = self.encoder.forward(
                word_ids_array_expanded, token_embeddings_array
            )

            # Size of counts: (unique_words)
            words_ids: list[int] = ids_mapping[:, 0].tolist()  # type: ignore[assignment]

            sentence_result: dict[str, WordEmbedding] = {}

            words = [self.vocab_resolver.lookup_word(word_id) for word_id in words_ids]

            for word, word_id, emb in zip(words, words_ids, minicoil_embeddings.tolist()):  # type: ignore[arg-type]
                if word_id == 0:
                    continue

                sentence_result[word] = WordEmbedding(
                    word=word,
                    forms=forms[word],
                    count=int(counts[word_id]),
                    word_id=int(word_id),
                    embedding=emb,  # type: ignore[arg-type]
                )

            for oov_word, count in oov.items():
                # {
                #     "word": oov_word,
                #     "forms": [oov_word],
                #     "count": int(count),
                #     "word_id": -1,
                #     "embedding": [1]
                # }
                sentence_result[oov_word] = WordEmbedding(
                    word=oov_word, forms=[oov_word], count=int(count), word_id=-1, embedding=[1]
                )

            if not is_query:
                yield self.sparse_vector_converter.embedding_to_vector(
                    sentence_result, vocab_size=vocab_size, embedding_size=embedding_size
                )
            else:
                yield self.sparse_vector_converter.embedding_to_vector_query(
                    sentence_result, vocab_size=vocab_size, embedding_size=embedding_size
                )

    @classmethod
    def _get_worker_class(cls) -> Type["MiniCoilTextEmbeddingWorker"]:
        return MiniCoilTextEmbeddingWorker


class MiniCoilTextEmbeddingWorker(TextEmbeddingWorker[SparseEmbedding]):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs: Any) -> MiniCOIL:
        return MiniCOIL(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
