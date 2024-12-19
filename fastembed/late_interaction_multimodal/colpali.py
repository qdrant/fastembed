from typing import Any, Iterable, Optional, Sequence, Type, Union

import numpy as np
from tokenizers import Encoding

from fastembed.common import OnnxProvider, ImageInput
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.utils import define_cache_dir
from fastembed.late_interaction_multimodal.late_interaction_multimodal_embedding_base import (
    LateInteractionMultimodalEmbeddingBase,
)
from fastembed.late_interaction_multimodal.onnx_multimodal_model import (
    OnnxMultimodalModel,
    TextEmbeddingWorker,
    ImageEmbeddingWorker,
)


supported_colbert_models = [
    {
        "model": "colpali",
        "dim": ...,
        "description": "Late interaction model",
        "license": "mit",
        "size_in_GB": 6.06,
        "sources": {
            "hf": "colpali",
        },
        "model_file": "model.onnx",
    },
]


class ColPali(LateInteractionMultimodalEmbeddingBase, OnnxMultimodalModel[np.ndarray]):
    DOCUMENT_MARKER_TOKEN_ID = 2
    QUERY_PREFIX = "Query: "
    BOS_TOKEN = "<s>"
    PAD_TOKEN = "<pad>"
    QUERY_MARKER_TOKEN_ID = [2, 9413]
    IMAGE_PLACEHOLDER_SIZE = (3, 448, 448)
    EMPTY_TEXT_PLACEHOLDER = np.array([257152] * 1024 + [2, 50721, 573, 2416, 235265, 108])
    EVEN_ATTENTION_MASK = np.array([1] * 1030)

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
            device_ids (Optional[list[int]], optional): The list of device ids to use for data parallel processing in
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

    @classmethod
    def list_supported_models(cls) -> list[dict[str, Any]]:
        """Lists the supported models.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing the model information.
        """
        return supported_colbert_models

    def load_onnx_model(self) -> None:
        self._load_onnx_model(
            model_dir=self._model_dir,
            model_file=self.model_description["model_file"],
            threads=self.threads,
            providers=self.providers,
            cuda=self.cuda,
            device_id=self.device_id,
        )

    def _post_process_onnx_image_output(
        self,
        output: OnnxOutputContext,
    ) -> Iterable[np.ndarray]:
        """
        Post-process the ONNX model output to convert it into a usable format.

        Args:
            output (OnnxOutputContext): The raw output from the ONNX model.

        Returns:
            Iterable[np.ndarray]: Post-processed output as NumPy arrays.
        """
        return output.model_output.astype(np.float32)

    def _post_process_onnx_text_output(
        self,
        output: OnnxOutputContext,
    ) -> Iterable[np.ndarray]:
        """
        Post-process the ONNX model output to convert it into a usable format.

        Args:
            output (OnnxOutputContext): The raw output from the ONNX model.

        Returns:
            Iterable[np.ndarray]: Post-processed output as NumPy arrays.
        """
        return output.model_output.astype(np.float32)

    def tokenize(self, documents: list[str], **_) -> list[Encoding]:
        texts_query: list[str] = []

        for query in documents:
            query = self.BOS_TOKEN + self.QUERY_PREFIX + query + self.PAD_TOKEN * 10
            query += "\n"

            texts_query.append(query)
        encoded = self.tokenizer.encode_batch(documents)
        return encoded

    def _preprocess_onnx_text_input(
        self, onnx_input: dict[str, np.ndarray], **kwargs
    ) -> dict[str, np.ndarray]:
        onnx_input["input_ids"] = np.array(
            [self.QUERY_MARKER_TOKEN_ID + input_ids[2:] for input_ids in onnx_input["input_ids"]]
        )
        return onnx_input

    def embed_text(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs,
    ) -> Iterable[np.ndarray]:
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
            **kwargs,
        )

    def embed_images(
        self,
        images: ImageInput,
        batch_size: int = 16,
        parallel: Optional[int] = None,
        **kwargs,
    ) -> Iterable[np.ndarray]:
        """
        Encode a list of images into list of embeddings.

        Args:
            images: Iterator of image paths or single image path to embed
            batch_size: Batch size for encoding -- higher values will use more memory, but be faster
            parallel:
                If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
                If 0, use all available cores.
                If None, don't use data-parallel processing, use default onnxruntime threading instead.

        Returns:
            List of embeddings, one per document
        """
        yield from self._embed_images(
            model_name=self.model_name,
            cache_dir=str(self.cache_dir),
            images=images,
            batch_size=batch_size,
            parallel=parallel,
            providers=self.providers,
            cuda=self.cuda,
            device_ids=self.device_ids,
            **kwargs,
        )

    @classmethod
    def _get_text_worker_class(cls) -> Type[TextEmbeddingWorker]:
        return ColPaliTextEmbeddingWorker

    @classmethod
    def _get_image_worker_class(cls) -> Type[ImageEmbeddingWorker]:
        return ColPaliImageEmbeddingWorker


class ColPaliTextEmbeddingWorker(TextEmbeddingWorker):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs) -> ColPali:
        return ColPali(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )


class ColPaliImageEmbeddingWorker(ImageEmbeddingWorker):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs) -> ColPali:
        return ColPali(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
