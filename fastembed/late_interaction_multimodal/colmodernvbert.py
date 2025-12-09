from typing import Any, Iterable, Type, Union, Optional, Sequence
import json

import numpy as np
from tokenizers import Encoding

from fastembed.common import ImageInput
from fastembed.common.model_description import DenseModelDescription, ModelSource
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.types import NumpyArray, OnnxProvider
from fastembed.common.utils import define_cache_dir
from fastembed.late_interaction_multimodal.late_interaction_multimodal_embedding_base import (
    LateInteractionMultimodalEmbeddingBase,
)
from fastembed.late_interaction_multimodal.onnx_multimodal_model import (
    OnnxMultimodalModel,
    TextEmbeddingWorker,
    ImageEmbeddingWorker,
)

supported_colmodernvbert_models: list[DenseModelDescription] = [
    DenseModelDescription(
        model="Qdrant/colmodernvbert",
        dim=128,
        description="The late-interaction version of ModernVBERT, CPU friendly, English, 2025.",
        license="mit",
        size_in_GB=1.0,
        sources=ModelSource(hf="Qdrant/colmodernvbert"),
        additional_files=["model.onnx_data"],
        model_file="model.onnx",
    ),
]


class ColModernVBERT(LateInteractionMultimodalEmbeddingBase, OnnxMultimodalModel[NumpyArray]):
    """
    The ModernVBERT/colmodernvbert model implementation. This model uses
    bidirectional attention, which proves to work better for retrieval.

    See: https://huggingface.co/ModernVBERT/colmodernvbert
    """

    VISUAL_PROMPT_PREFIX = (
        "<|begin_of_text|>User:<image>Describe the image.<end_of_utterance>\nAssistant:"
    )

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
        self._extra_session_options = self._select_exposed_session_options(kwargs)

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
        self.image_seq_len: Optional[int] = None
        self.max_image_size: Optional[int] = None
        self.image_size: Optional[int] = None

        if not self.lazy_load:
            self.load_onnx_model()

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        """Lists the supported models.

        Returns:
            list[DenseModelDescription]: A list of DenseModelDescription objects containing the model information.
        """
        return supported_colmodernvbert_models

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

        # Load image processing configuration
        processor_config_path = self._model_dir / "processor_config.json"
        with open(processor_config_path) as f:
            processor_config = json.load(f)
            self.image_seq_len = processor_config.get("image_seq_len", 64)

        preprocessor_config_path = self._model_dir / "preprocessor_config.json"
        with open(preprocessor_config_path) as f:
            preprocessor_config = json.load(f)
            self.max_image_size = preprocessor_config.get("max_image_size", {}).get(
                "longest_edge", 512
            )

        # Load model configuration
        config_path = self._model_dir / "config.json"
        with open(config_path) as f:
            model_config = json.load(f)
            vision_config = model_config.get("vision_config", {})
            self.image_size = vision_config.get("image_size", 512)

    def _preprocess_onnx_text_input(
        self, onnx_input: dict[str, NumpyArray], **kwargs: Any
    ) -> dict[str, NumpyArray]:
        """
        Post-process the ONNX model output to convert it into a usable format.

        Args:
            output (OnnxOutputContext): The raw output from the ONNX model.

        Returns:
            Iterable[NumpyArray]: Post-processed output as NumPy arrays.
        """
        batch_size, seq_length = onnx_input["input_ids"].shape
        empty_image_placeholder: NumpyArray = np.zeros(
            (batch_size, seq_length, 3, self.image_size, self.image_size), dtype=np.float32  # type: ignore[type-var,arg-type,assignment]
        )
        onnx_input["pixel_values"] = empty_image_placeholder
        return onnx_input

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
        encoded = self.tokenizer.encode_batch(documents)  # type: ignore[union-attr]
        return encoded

    def _preprocess_onnx_image_input(
        self, onnx_input: dict[str, np.ndarray], **kwargs: Any
    ) -> dict[str, NumpyArray]:
        """
        Add text input placeholders for image data, following Idefics3 processing logic.

        Constructs input_ids dynamically based on the actual number of image patches,
        using the same token expansion logic as Idefics3Processor.

        Args:
            onnx_input: Dict with 'pixel_values' (batch, num_patches, C, H, W)
                        and 'attention_mask' (batch, num_patches) indicating real patches
            **kwargs: Additional arguments

        Returns:
            Updated onnx_input with 'input_ids' and updated 'attention_mask' for token sequence
        """
        # The attention_mask in onnx_input has a shape of (batch_size, num_patches),
        # and should be used to create an attention mask matching the input_ids shape.
        patch_attention_mask = onnx_input["attention_mask"]
        pixel_values = onnx_input["pixel_values"]

        batch_size = pixel_values.shape[0]
        batch_input_ids = []

        # Build input_ids for each image based on its actual patch count
        for i in range(batch_size):
            # Count real patches (non-padded) from attention mask
            patch_count = int(np.sum(patch_attention_mask[i]))

            # Compute rows/cols from patch count
            rows, cols = self._compute_rows_cols_from_patches(patch_count)

            # Build input_ids for this image
            input_ids = self._build_input_ids_for_image(rows, cols)
            batch_input_ids.append(input_ids)

        # Pad sequences to max length in batch
        max_len = max(len(ids) for ids in batch_input_ids)

        # Get padding config from tokenizer
        padding_direction = self.tokenizer.padding["direction"]  # type: ignore[index,union-attr]
        pad_token_id = self.tokenizer.padding["pad_id"]  # type: ignore[index,union-attr]

        # Initialize with pad token
        padded_input_ids = np.full((batch_size, max_len), pad_token_id, dtype=np.int64)
        attention_mask = np.zeros((batch_size, max_len), dtype=np.int64)

        for i, input_ids in enumerate(batch_input_ids):
            seq_len = len(input_ids)
            if padding_direction == "left":
                # Left padding: place tokens at the END of the array
                start_idx = max_len - seq_len
                padded_input_ids[i, start_idx:] = input_ids
                attention_mask[i, start_idx:] = 1
            else:
                # Right padding: place tokens at the START of the array
                padded_input_ids[i, :seq_len] = input_ids
                attention_mask[i, :seq_len] = 1

        onnx_input["input_ids"] = padded_input_ids
        # Update attention_mask with token-level data
        onnx_input["attention_mask"] = attention_mask
        return onnx_input

    def _compute_rows_cols_from_patches(self, patch_count: int) -> tuple[int, int]:
        if patch_count <= 1:
            return 0, 0

        # Subtract 1 for the global image
        grid_patches = patch_count - 1

        # Find rows and cols (assume square or near-square grid)
        rows = int(grid_patches**0.5)
        cols = grid_patches // rows

        # Verify the calculation
        if rows * cols + 1 != patch_count:
            # Handle non-square grids
            for r in range(1, grid_patches + 1):
                if grid_patches % r == 0:
                    c = grid_patches // r
                    if r * c + 1 == patch_count:
                        return r, c
            # Fallback: treat as unsplit
            return 0, 0

        return rows, cols

    def _create_single_image_prompt_string(self) -> str:
        return (
            "<fake_token_around_image>"
            + "<global-img>"
            + "<image>" * self.image_seq_len  # type: ignore[operator]
            + "<fake_token_around_image>"
        )

    def _create_split_image_prompt_string(self, rows: int, cols: int) -> str:
        text_split_images = ""

        # Add tokens for each patch in the grid
        for n_h in range(rows):
            for n_w in range(cols):
                text_split_images += (
                    "<fake_token_around_image>"
                    + f"<row_{n_h + 1}_col_{n_w + 1}>"
                    + "<image>" * self.image_seq_len  # type: ignore[operator]
                )
            text_split_images += "\n"

        # Add global image at the end
        text_split_images += (
            "\n<fake_token_around_image>"
            + "<global-img>"
            + "<image>" * self.image_seq_len  # type: ignore[operator]
            + "<fake_token_around_image>"
        )

        return text_split_images

    def _build_input_ids_for_image(self, rows: int, cols: int) -> np.ndarray:
        # Create the appropriate image prompt string
        if rows == 0 and cols == 0:
            image_prompt_tokens = self._create_single_image_prompt_string()
        else:
            image_prompt_tokens = self._create_split_image_prompt_string(rows, cols)

        # Replace <image> in visual prompt with expanded tokens
        # The visual prompt is: "<|begin_of_text|>User:<image>Describe the image.<end_of_utterance>\nAssistant:"
        expanded_prompt = self.VISUAL_PROMPT_PREFIX.replace("<image>", image_prompt_tokens)

        # Tokenize the complete prompt
        encoded = self.tokenizer.encode(expanded_prompt)  # type: ignore[union-attr]

        # Convert to numpy array
        return np.array(encoded.ids, dtype=np.int64)

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
            extra_session_options=self._extra_session_options,
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
            extra_session_options=self._extra_session_options,
            **kwargs,
        )

    @classmethod
    def _get_text_worker_class(cls) -> Type[TextEmbeddingWorker[NumpyArray]]:
        return ColModernVBERTTextEmbeddingWorker

    @classmethod
    def _get_image_worker_class(cls) -> Type[ImageEmbeddingWorker[NumpyArray]]:
        return ColModernVBERTImageEmbeddingWorker


class ColModernVBERTTextEmbeddingWorker(TextEmbeddingWorker[NumpyArray]):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs: Any) -> ColModernVBERT:
        return ColModernVBERT(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )


class ColModernVBERTImageEmbeddingWorker(ImageEmbeddingWorker[NumpyArray]):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs: Any) -> ColModernVBERT:
        return ColModernVBERT(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
