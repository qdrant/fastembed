from typing import Iterable, Optional, Any, Union

from fastembed.common.model_description import DenseModelDescription
from fastembed.common.types import NumpyArray
from fastembed.common.model_management import ModelManagement
from fastembed.common.types import ImageInput


class ImageEmbeddingBase(ModelManagement[DenseModelDescription]):
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

    def embed(
        self,
        images: Union[ImageInput, Iterable[ImageInput]],
        batch_size: int = 16,
        parallel: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        """
        Embeds a list of images into a list of embeddings.

        Args:
            images: The list of image paths to preprocess and embed.
            batch_size: Batch size for encoding
            parallel:
                If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
                If 0, use all available cores.
                If None, don't use data-parallel processing, use default onnxruntime threading instead.
            **kwargs: Additional keyword argument to pass to the embed method.

        Yields:
            Iterable[NdArray]: The embeddings.
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
