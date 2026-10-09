import contextlib
from typing import Any, Iterable, Type, Optional, Sequence
import json

import numpy as np
import onnxruntime as ort
from tokenizers import Encoding
from PIL import Image

from fastembed.common import ImageInput
from fastembed.common.model_description import DenseModelDescription, ModelSource
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.types import NumpyArray, OnnxProvider
from fastembed.common.utils import define_cache_dir, iter_batch
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
        additional_files=["processor_config.json"],
        model_file="model_v2.onnx",
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
    QUERY_AUGMENTATION_TOKEN = "<end_of_utterance>"

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
        # onnxruntime 1.18 crashes the whole process while optimizing this graph (an access violation
        # on Windows, a failed bounds check on Linux), and 1.17 can't read its IR version
        if tuple(int(part) for part in ort.__version__.split(".")[:2]) < (1, 19):
            raise RuntimeError(
                f"Could not load {self.model_name}: it requires onnxruntime>=1.19, but "
                f"onnxruntime {ort.__version__} is installed. Please upgrade onnxruntime."
            )
        self._load_onnx_model(
            model_dir=self._model_dir,
            model_file=self.model_description.model_file,
            threads=self.threads,
            providers=self.providers,
            cuda=self.cuda,
            device_id=self.device_id,
            extra_session_options=self._extra_session_options,
            additional_files=self.model_description.additional_files,
        )

        # Load image processing configuration
        processor_config_path = self._model_dir / "processor_config.json"
        with open(processor_config_path, encoding="utf-8") as f:
            processor_config = json.load(f)
            self.image_seq_len = processor_config.get("image_seq_len", 64)

        preprocessor_config_path = self._model_dir / "preprocessor_config.json"
        with open(preprocessor_config_path, encoding="utf-8") as f:
            preprocessor_config = json.load(f)
            self.max_image_size = preprocessor_config.get("max_image_size", {}).get(
                "longest_edge", 512
            )

        # Load model configuration
        config_path = self._model_dir / "config.json"
        with open(config_path, encoding="utf-8") as f:
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
        # one blank tile for the whole batch: the model drops blank tiles and skips the vision
        # encoder; zero tiles would also skip it, but CUDA can't reduce the resulting empty tensor
        blank_image_placeholder: NumpyArray = np.zeros(
            (1, 1, 3, self.image_size, self.image_size),
            dtype=np.float32,  # type: ignore[type-var,arg-type,assignment]
        )
        onnx_input["pixel_values"] = blank_image_placeholder
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
        assert output.attention_mask is not None
        # drop the padding rows, so a query gets the same vectors whatever else is in its batch
        for embedding, attention_mask in zip(output.model_output, output.attention_mask):
            yield embedding[attention_mask == 1]

    def tokenize(self, documents: list[str], **kwargs: Any) -> list[Encoding]:
        # Add query augmentation tokens (matching process_queries logic from colpali-engine)
        augmented_queries = [doc + self.QUERY_AUGMENTATION_TOKEN * 10 for doc in documents]
        encoded = self.tokenizer.encode_batch(augmented_queries)  # type: ignore[union-attr]
        return encoded

    def token_count(
        self,
        texts: str | Iterable[str],
        batch_size: int = 1024,
        include_extension: bool = False,
        **kwargs: Any,
    ) -> int:
        self._ensure_tokenizer()
        token_num = 0
        texts = [texts] if isinstance(texts, str) else texts
        assert self.tokenizer is not None
        tokenize_func = self.tokenize if include_extension else self.tokenizer.encode_batch
        for batch in iter_batch(texts, batch_size):
            token_num += sum([sum(encoding.attention_mask) for encoding in tokenize_func(batch)])
        return token_num

    def onnx_embed_image(self, images: list[ImageInput], **kwargs: Any) -> OnnxOutputContext:
        with contextlib.ExitStack() as stack:
            image_files = [
                stack.enter_context(Image.open(image))
                if not isinstance(image, Image.Image)
                else image
                for image in images
            ]
            assert self.processor is not None, "Processor is not initialized"
            processor_metadata: dict[str, Any] = {}
            processed = self.processor(image_files, metadata=processor_metadata)
            encoded, attention_mask, metadata = self._process_nested_patches(processed)  # type: ignore[arg-type]
            metadata.update(processor_metadata)

        onnx_input = {"pixel_values": encoded, "attention_mask": attention_mask}
        onnx_input = self._preprocess_onnx_image_input(
            onnx_input, image_grid=metadata["image_grid"], **kwargs
        )
        model_output = self.model.run(None, onnx_input)  # type: ignore[union-attr]

        return OnnxOutputContext(
            model_output=model_output[0],
            # the token mask, not the tile mask: post-processing drops the padding rows with it
            attention_mask=onnx_input["attention_mask"],  # type: ignore[arg-type]
            metadata=metadata,
        )

    @staticmethod
    def _process_nested_patches(
        processed: list[list[NumpyArray]],
    ) -> tuple[NumpyArray, NumpyArray, dict[str, Any]]:
        """
        Process nested image patches (from ImageSplitter).

        Args:
            processed: List of patch lists, one per image [[img1_patches], [img2_patches], ...]

        Returns:
            tuple: (encoded array, attention_mask, metadata)
                - encoded: (batch_size, max_patches, C, H, W)
                - attention_mask: (batch_size, max_patches) with 1 for real patches, 0 for padding
                - metadata: Dict with 'patch_counts' key
        """
        patch_counts = [len(patches) for patches in processed]
        max_patches = max(patch_counts)

        # Get dimensions from first patch
        channels, height, width = processed[0][0].shape
        batch_size = len(processed)

        # Create padded array
        encoded = np.zeros(
            (batch_size, max_patches, channels, height, width), dtype=processed[0][0].dtype
        )

        # Create attention mask (1 for real patches, 0 for padding)
        attention_mask = np.zeros((batch_size, max_patches), dtype=np.int64)

        # Fill in patches and attention mask
        for i, patches in enumerate(processed):
            for j, patch in enumerate(patches):
                encoded[i, j] = patch
                attention_mask[i, j] = 1

        metadata = {"patch_counts": patch_counts}
        return encoded, attention_mask, metadata  # type: ignore[return-value]

    def _preprocess_onnx_image_input(
        self,
        onnx_input: dict[str, np.ndarray],
        *,
        image_grid: list[tuple[int, int]] | None = None,
        **kwargs: Any,
    ) -> dict[str, NumpyArray]:
        """
        Add text input placeholders for image data, following Idefics3 processing logic.

        Constructs input_ids from the actual image patch grid, using the same
        token expansion logic as Idefics3Processor.

        Args:
            onnx_input: Dict with 'pixel_values' (batch, num_patches, C, H, W)
                        and 'attention_mask' (batch, num_patches) indicating real patches
            image_grid: Actual (rows, cols) from image splitting, (0, 0) if unsplit
            **kwargs: Additional arguments

        Returns:
            Updated onnx_input with 'input_ids' and updated 'attention_mask' for token sequence
        """
        # The attention_mask in onnx_input has a shape of (batch_size, num_patches),
        # and should be used to create an attention mask matching the input_ids shape.
        patch_attention_mask = onnx_input["attention_mask"]
        pixel_values = onnx_input["pixel_values"]

        batch_size = pixel_values.shape[0]
        if image_grid is None or len(image_grid) != batch_size:
            raise ValueError("The image patch grid is required for each image in the batch")
        batch_input_ids = []

        # A patch count cannot distinguish portrait, landscape, and square grids.
        for i, (rows, cols) in enumerate(image_grid):
            # Count real patches (non-padded) from attention mask
            patch_count = int(np.sum(patch_attention_mask[i]))

            valid_grid = (rows == 0 and cols == 0) or (rows > 0 and cols > 0)
            if not valid_grid or rows * cols + 1 != patch_count:
                raise ValueError("The image patch grid does not match the number of patches")

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
        assert output.attention_mask is not None
        embeddings: NumpyArray = output.model_output.reshape(
            output.model_output.shape[0], -1, self.model_description.dim
        )
        # drop the padding rows, so an image gets the same vectors whatever else is in its batch
        for embedding, attention_mask in zip(embeddings, output.attention_mask):
            yield embedding[attention_mask == 1]

    def embed_text(
        self,
        documents: str | Iterable[str],
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
        images: ImageInput | Iterable[ImageInput],
        batch_size: int = 2,
        parallel: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        """
        Encode a list of images into list of embeddings.

        Args:
            images: Iterator of image paths or single image path to embed
            batch_size: Batch size for encoding -- every image is split into up to 17 tiles, so
                memory grows quickly with it, while on CPU larger batches are not faster
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
