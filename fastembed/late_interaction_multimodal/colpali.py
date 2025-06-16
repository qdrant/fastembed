from typing import Any, Iterable, Optional, Sequence, Type, Union

import numpy as np
from tokenizers import Encoding

from fastembed.common import OnnxProvider, ImageInput
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.types import NumpyArray
from fastembed.common.utils import define_cache_dir
from fastembed.late_interaction_multimodal.late_interaction_multimodal_embedding_base import (
    LateInteractionMultimodalEmbeddingBase,
)
from fastembed.late_interaction_multimodal.onnx_multimodal_model import (
    OnnxMultimodalModel,
    TextEmbeddingWorker,
    ImageEmbeddingWorker,
)
from fastembed.common.model_description import DenseModelDescription, ModelSource

supported_colpali_models: list[DenseModelDescription] = [
    DenseModelDescription(
        model="Qdrant/colpali-v1.3-fp16",
        dim=128,
        description="Text embeddings, Multimodal (text&image), English, 50 tokens query length truncation, 2024.",
        license="mit",
        size_in_GB=6.5,
        sources=ModelSource(hf="Qdrant/colpali-v1.3-fp16"),
        additional_files=["model.onnx_data"],
        model_file="model.onnx",
    ),
]


class ColPali(LateInteractionMultimodalEmbeddingBase, OnnxMultimodalModel[NumpyArray]):
    QUERY_PREFIX = "Query: "
    BOS_TOKEN = "<s>"
    PAD_TOKEN = "<pad>"
    QUERY_MARKER_TOKEN_ID = [2, 5098]
    IMAGE_PLACEHOLDER_SIZE = (3, 448, 448)
    EMPTY_TEXT_PLACEHOLDER = np.array(
        [257152] * 1024 + [2, 50721, 573, 2416, 235265, 108]
    )  # This is a tokenization of '<image>' * 1024 + '<bos>Describe the image.\n' line which is used as placeholder
    # while processing an image
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
        self.mask_token_id = None
        self.pad_token_id = None

        if not self.lazy_load:
            self.load_onnx_model()

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        """Lists the supported models.

        Returns:
            list[DenseModelDescription]: A list of DenseModelDescription objects containing the model information.
        """
        return supported_colpali_models

    def load_onnx_model(self) -> None:
        self._load_onnx_model(
            model_dir=self._model_dir,
            model_file=self.model_description.model_file,
            threads=self.threads,
            providers=self.providers,
            cuda=self.cuda,
            device_id=self.device_id,
        )

    def _post_process_onnx_image_output(
        self,
        output: OnnxOutputContext,
    ) -> Iterable[NumpyArray]:
        """
        Post-process the ONNX model output to convert it into a usable format.

        Args:
            output (OnnxOutputContext): The raw output from the ONNX model.

        Returns:
            Iterable[NumpyArray]: Post-processed output as NumPy arrays.
        """
        assert self.model_description.dim is not None, "Model dim is not defined"
        return output.model_output.reshape(
            output.model_output.shape[0], -1, self.model_description.dim
        )

    def _post_process_onnx_text_output(
        self,
        output: OnnxOutputContext,
    ) -> Iterable[NumpyArray]:
        """
        Post-process the ONNX model output to convert it into a usable format.

        Args:
            output (OnnxOutputContext): The raw output from the ONNX model.

        Returns:
            Iterable[NumpyArray]: Post-processed output as NumPy arrays.
        """
        return output.model_output

    def tokenize(self, documents: list[str], **kwargs: Any) -> list[Encoding]:
        texts_query: list[str] = []
        for query in documents:
            query = self.BOS_TOKEN + self.QUERY_PREFIX + query + self.PAD_TOKEN * 10
            query += "\n"

            texts_query.append(query)
        encoded = self.tokenizer.encode_batch(texts_query)  # type: ignore[union-attr]
        return encoded

    def _preprocess_onnx_text_input(
        self, onnx_input: dict[str, NumpyArray], **kwargs: Any
    ) -> dict[str, NumpyArray]:
        onnx_input["input_ids"] = np.array(
            [
                self.QUERY_MARKER_TOKEN_ID + input_ids[2:].tolist()  # type: ignore[index]
                for input_ids in onnx_input["input_ids"]
            ]
        )
        empty_image_placeholder: NumpyArray = np.zeros(
            self.IMAGE_PLACEHOLDER_SIZE, dtype=np.float32
        )
        onnx_input["pixel_values"] = np.array(
            [empty_image_placeholder for _ in onnx_input["input_ids"]],
        )
        return onnx_input

    def _preprocess_onnx_image_input(
        self, onnx_input: dict[str, np.ndarray], **kwargs: Any
    ) -> dict[str, NumpyArray]:
        """
        Add placeholders for text input when processing image data for ONNX.
        Args:
            onnx_input (Dict[str, NumpyArray]): Preprocessed image inputs.
            **kwargs: Additional arguments.
        Returns:
            Dict[str, NumpyArray]: ONNX input with text placeholders.
        """
        onnx_input["input_ids"] = np.array(
            [self.EMPTY_TEXT_PLACEHOLDER for _ in onnx_input["pixel_values"]]
        )
        onnx_input["attention_mask"] = np.array(
            [self.EVEN_ATTENTION_MASK for _ in onnx_input["pixel_values"]]
        )
        return onnx_input

    def embed_text(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
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
            **kwargs,
        )

    def embed_image(
        self,
        images: Union[ImageInput, Iterable[ImageInput]],
        batch_size: int = 16,
        parallel: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
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
            local_files_only=self._local_files_only,
            specific_model_path=self._specific_model_path,
            **kwargs,
        )

    @classmethod
    def _get_text_worker_class(cls) -> Type[TextEmbeddingWorker[NumpyArray]]:
        return ColPaliTextEmbeddingWorker

    @classmethod
    def _get_image_worker_class(cls) -> Type[ImageEmbeddingWorker[NumpyArray]]:
        return ColPaliImageEmbeddingWorker


class ColPaliTextEmbeddingWorker(TextEmbeddingWorker[NumpyArray]):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs: Any) -> ColPali:
        return ColPali(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )


class ColPaliImageEmbeddingWorker(ImageEmbeddingWorker[NumpyArray]):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs: Any) -> ColPali:
        return ColPali(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
