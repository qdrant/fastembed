from typing import Iterable, Optional, Union

import numpy as np

from fastembed.common import ImageInput
from fastembed.common.model_management import ModelManagement


class LateInteractionMultimodalEmbeddingBase(ModelManagement):
    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.threads = threads
        self._local_files_only = kwargs.pop("local_files_only", False)

    def embed_text(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs,
    ) -> Iterable[np.ndarray]:
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
            Iterable[np.ndarray]: The embeddings.
        """
        raise NotImplementedError()

    def embed_image(
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
            List of embeddings, one per image
        """
        raise NotImplementedError()
