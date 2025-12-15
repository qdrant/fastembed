from typing import Iterable, Optional, Union, Any


from fastembed.common import ImageInput
from fastembed.common.model_description import DenseModelDescription
from fastembed.common.model_management import ModelManagement
from fastembed.common.types import NumpyArray


class LateInteractionMultimodalEmbeddingBase(ModelManagement[DenseModelDescription]):
    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.threads = threads
        self._local_files_only = kwargs.pop("local_files_only", False)
        self._embedding_size: Optional[int] = None

    def embed_text(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        """
        Embeds a list of documents into a list of embeddings.

        Args:
            documents (Iterable[str]): The list of texts to embed.
            batch_size: Batch size for encoding -- higher values will use more memory, but be faster
            parallel:
                If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
                If 0, use all available cores.
                If None, don't use data-parallel processing, use default onnxruntime threading instead.
            **kwargs: Additional keyword argument to pass to the embed method.

        Yields:
            Iterable[NumpyArray]: The embeddings.
        """
        raise NotImplementedError()

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
            List of embeddings, one per image
        """
        raise NotImplementedError()

    @classmethod
    def get_embedding_size(cls, model_name: str) -> int:
        """Returns embedding size of the chosen model."""
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def embedding_size(self) -> int:
        """Returns embedding size for the current model"""
        raise NotImplementedError("Subclasses must implement this method")

    def get_image_mask(
        self,
        images: Union[ImageInput, Iterable[ImageInput]],
        batch_size: int = 16,
        **kwargs: Any,
    ) -> list[NumpyArray]:
        """
        Generate binary masks identifying image tokens in processed image sequences.

        This method processes images and returns masks indicating which tokens in the
        resulting sequence correspond to image content (value=1) vs text/special tokens (value=0).

        Args:
            images: Single image or iterable of images (file paths, bytes, or PIL Image objects)
            batch_size: Number of images to process in each batch. Defaults to 16.
            **kwargs: Additional keyword arguments for image processing.

        Returns:
            List of binary masks (numpy arrays with dtype=bool), one per image. Each mask has shape (sequence_length,)
            where sequence_length is the number of tokens in the processed image representation.
            Values are True for image tokens, False for non-image tokens (text, special tokens, etc.).

        Raises:
            NotImplementedError: If the model doesn't support image mask generation.

        Example:
            ```python
            model = ColPali.load("Qdrant/colpali-v1.3-fp16")
            masks = model.get_image_mask(["image1.jpg", "image2.jpg"])
            # masks[0] is a numpy array of shape (1030,) with dtype=bool for ColPali
            # First 1024 values are True (image tokens), last 6 are False (text tokens)
            ```
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support image mask generation. "
            "Override this method in subclasses to provide model-specific implementation."
        )
