from typing import Any, Iterable, Optional, Sequence, Union, List, Dict, Type

import numpy as np
from sys import maxsize
from fastembed.common import OnnxProvider
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.image.onnx_image_model import OnnxImageModel
from fastembed.late_interaction.late_interaction_image_embedding_base import (
    LateInteractionImageEmbeddingBase,
)
from PIL import Image
from fastembed.text.onnx_text_model import OnnxTextModel, TextEmbeddingWorker
import contextlib
from fastembed.common import ImageInput
from fastembed.common.preprocessor_utils import load_preprocessor


supported_colpali_models = [
    {
        "model": "akshayballal/colpali-v1.2-merged",
        "dim": 128,
        "description": "Text embeddings, Unimodal (text), Aligned to image latent space, ColBERT-compatible, 512 tokens max, 2024.",
        "license": "mit",
        "size_in_GB": 6.08,
        "sources": {
            "hf": "akshayballal/colpali-v1.2-merged-onnx",
        },
        "additional_files": [
            "model.onnx_data",
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
        ],
        "model_file": "model.onnx",
    }
]


class ColPali(
    LateInteractionImageEmbeddingBase, OnnxTextModel[np.ndarray], OnnxImageModel[np.array]
):
    DOCUMENT_MARKER_TOKEN_ID = 2
    QUERY_PREFIX = "Query: "
    BOS_TOKEN = "<s>"
    PAD_TOKEN = "<pad>"
    QUERY_MARKER_TOKEN_ID = [2, 9413]
    image_placeholder_size = (3, 448, 448)
    EMPTY_TEXT_PLACEHOLDER = np.array([257152] * 1024 + [2, 50721, 573, 2416, 235265, 108])
    EVEN_ATTENTION_MASK = np.array([1] * 1030)

    def _post_process_onnx_output(
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

    @classmethod
    def list_supported_models(cls) -> list[dict[str, Any]]:
        """
        Lists the supported models.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing the model information.
        """
        return supported_colpali_models

    def _preprocess_queries(self, documents: list[str]) -> list[str]:
        """
        Preprocess the input text queries by adding special tokens and padding.

        Args:
            documents (list[str]): List of text queries.

        Returns:
            list[str]: Preprocessed text queries.
        """
        texts_query: list[str] = []

        for query in documents:
            query = self.BOS_TOKEN + self.QUERY_PREFIX + query + self.PAD_TOKEN * 10
            query += "\n"

            texts_query.append(query)
        return texts_query

    def _preprocess_images_input(
        self, inputs: list[Union[ImageInput]], **kwargs: Any
    ) -> dict[str, np.ndarray]:
        """
        Preprocess the input images for ONNX model inference.

        Args:
            inputs (list[Union[ImageInput]]): List of image inputs.
            **kwargs: Additional preprocessing arguments.

        Returns:
            dict[str, np.ndarray]: Preprocessed image inputs as a dictionary.
        """
        with contextlib.ExitStack():
            image_files = [
                Image.open(image) if not isinstance(image, Image.Image) else image
                for image in inputs
            ]
            encoded = self.processor(image_files)
            onnx_input = self._build_onnx_input(encoded)
            onnx_input = self._preprocess_image_input(onnx_input, **kwargs)
            return onnx_input

    def embed(
        self,
        inputs: Union[ImageInput, str],
        batch_size: int = 16,
        parallel: Optional[int] = None,
        is_doc: bool = False,
        **kwargs,
    ) -> OnnxOutputContext:
        """
        Generate embeddings for the given input, either images or text.

        Args:
            inputs (Union[ImageInput, str]): Input data (images or text).
            batch_size (int, optional): Batch size for embedding. Defaults to 16.
            parallel (Optional[int], optional): Number of parallel threads. Defaults to None.
            is_doc (bool, optional): Indicates if input is a document. Defaults to False.
            **kwargs: Additional arguments for embedding.

        Yields:
            OnnxOutputContext: Embedding output context.
        """
        if is_doc:
            yield from self._embed_documents(
                model_name=self.model_name,
                cache_dir=str(self.cache_dir),
                documents=inputs,
                batch_size=batch_size,
                parallel=parallel,
                providers=self.providers,
                cuda=self.cuda,
                device_ids=self.device_ids,
                **kwargs,
            )
        else:
            # onnx_input = self._preprocess_images_input(inputs, **kwargs)
            yield from self._embed_images(
                model_name=self.model_name,
                cache_dir=str(self.cache_dir),
                images=inputs,
                batch_size=batch_size,
                parallel=parallel,
                providers=self.providers,
                cuda=self.cuda,
                device_ids=self.device_ids,
                **kwargs,
            )

    def onnx_embed(self, inputs: Union[ImageInput, str], **kwargs) -> OnnxOutputContext:
        """
        Embed inputs using the ONNX model.

        Args:
            inputs (Union[ImageInput, str]): Input data (images or text).
            **kwargs: Additional arguments for embedding.

        Returns:
            OnnxOutputContext: Embedding output context.
        """
        if isinstance(inputs[0], str):
            return self.onnx_embed_text(inputs, **kwargs)
        else:
            return self.onnx_embed_image(inputs, **kwargs)

    def onnx_embed_image(self, images: List[ImageInput], **kwargs) -> OnnxOutputContext:
        """
        Embed images using the ONNX model.

        Args:
            images (List[ImageInput]): List of image inputs.
            **kwargs: Additional arguments for embedding.

        Returns:
            OnnxOutputContext: Embedding output context for images.
        """
        with contextlib.ExitStack():
            image_files = [
                Image.open(image) if not isinstance(image, Image.Image) else image
                for image in images
            ]
            encoded = self.processor(image_files)
        onnx_input = self._build_onnx_input(encoded)
        onnx_input = self._preprocess_onnx_image_input(onnx_input)
        model_output = self.model.run(None, onnx_input)
        embeddings = model_output[0].reshape(len(images), -1, self.model_description["dim"])
        return OnnxOutputContext(model_output=embeddings)

    def onnx_embed_text(
        self,
        documents: List[str],
        **kwargs,
    ) -> OnnxOutputContext:
        """
        Embed text using the ONNX model.

        Args:
            documents (List[str]): List of text documents.
            **kwargs: Additional arguments for embedding.

        Returns:
            OnnxOutputContext: Embedding output context for text.
        """
        documents = self._preprocess_queries(documents)
        encoded = self.tokenize(documents, **kwargs)
        input_ids = np.array([self.QUERY_MARKER_TOKEN_ID + e.ids[2:] for e in encoded])

        attention_mask = np.array([e.attention_mask for e in encoded])
        onnx_input = {"input_ids": np.array(input_ids, dtype=np.int64)}
        onnx_input = self._preprocess_onnx_text_input(onnx_input, **kwargs)
        onnx_input["attention_mask"] = attention_mask
        model_output = self.model.run(self.ONNX_OUTPUT_NAMES, onnx_input)
        return OnnxOutputContext(
            model_output=model_output[0],
            attention_mask=onnx_input.get("attention_mask", attention_mask),
            input_ids=onnx_input.get("input_ids", input_ids),
        )

    def _preprocess_onnx_image_input(
        self, onnx_input: Dict[str, np.ndarray], **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Add placeholders for text input when processing image data for ONNX.

        Args:
            onnx_input (Dict[str, np.ndarray]): Preprocessed image inputs.
            **kwargs: Additional arguments.

        Returns:
            Dict[str, np.ndarray]: ONNX input with text placeholders.
        """
        onnx_input["input_ids"] = np.array(
            [self.EMPTY_TEXT_PLACEHOLDER for _ in onnx_input["input_ids"]]
        )
        onnx_input["attention_mask"] = np.array(
            [self.EVEN_ATTENTION_MASK for _ in onnx_input["input_ids"]]
        )
        return onnx_input

    def _preprocess_onnx_text_input(
        self, onnx_input: Dict[str, np.ndarray], **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Add placeholders for image input when processing text data for ONNX.

        Args:
            onnx_input (Dict[str, np.ndarray]): Preprocessed text inputs.
            **kwargs: Additional arguments.

        Returns:
            Dict[str, np.ndarray]: ONNX input with image placeholders.
        """
        empty_image_placeholder = np.zeros(self.image_placeholder_size, dtype=np.float32)
        onnx_input["pixel_values"] = np.array(
            [empty_image_placeholder for _ in onnx_input["input_ids"]]
        )
        onnx_input["attention_mask"] = np.array([[1] for _ in onnx_input["input_ids"]])
        return onnx_input

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
        Initialize the ColPali model.

        Args:
            model_name (str): Name of the model to use.
            cache_dir (Optional[str], optional): Directory for caching model files. Defaults to None.
            threads (Optional[int], optional): Number of threads for inference. Defaults to None.
            providers (Optional[Sequence[OnnxProvider]], optional): ONNX providers for model execution. Defaults to None.
            cuda (bool, optional): Whether to use CUDA for inference. Defaults to False.
            device_ids (Optional[list[int]], optional): List of CUDA device IDs. Defaults to None.
            lazy_load (bool, optional): Whether to lazily load the model. Defaults to False.
            device_id (Optional[int], optional): Specific device ID to use. Defaults to None.
            **kwargs: Additional arguments for model initialization.
        """
        super().__init__(model_name, cache_dir, threads, **kwargs)
        self.model_description = self._get_model_description(model_name)
        self._model_dir = self.download_model(
            self.model_description, self.cache_dir, local_files_only=self._local_files_only
        )
        self.providers = providers
        self.lazy_load = lazy_load
        self.cuda = cuda
        self.device_ids = device_ids
        if device_id is not None:
            self.device_id = device_id
        elif self.device_ids is not None:
            self.device_id = self.device_ids[0]
        else:
            self.device_id = None
        if not self.lazy_load:
            self.load_onnx_model()

        self.processor = load_preprocessor(model_dir=self._model_dir)

    def load_onnx_model(self) -> None:
        """
        Load the ONNX model for inference.
        """
        self._load_onnx_model(
            model_dir=self._model_dir,
            model_file=self.model_description["model_file"],
            threads=self.threads,
            providers=self.providers,
            cuda=self.cuda,
            device_id=self.device_id,
        )
        self.tokenizer.enable_truncation(max_length=maxsize)

    @classmethod
    def _get_worker_class(cls) -> Type[TextEmbeddingWorker]:
        """
        Get the worker class for text/image embedding.

        Returns:
            Type[TextEmbeddingWorker]: The worker class.
        """
        return ColPaliEmbeddingWorker


class ColPaliEmbeddingWorker(TextEmbeddingWorker):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs) -> ColPali:
        """
        Initialize the ColPali embedding worker.

        Args:
            model_name (str): Name of the model to use.
            cache_dir (str): Directory for caching model files.
            **kwargs: Additional arguments for initialization.

        Returns:
            ColPali: Initialized ColPali model instance.
        """
        return ColPali(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
