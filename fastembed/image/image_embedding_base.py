from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np

from fastembed.common.model_management import ModelManagement


class ImageEmbeddingBase(ModelManagement):
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

    def embed(
        self,
        images: Union[Union[str, Path], Iterable[Union[str, Path]]],
        batch_size: int = 16,
        parallel: Optional[int] = None,
        **kwargs,
    ) -> Iterable[np.ndarray]:
        """
        Embeds a list of images into a list of embeddings.

        Args:
            images - The list of image paths to preprocess and embed.
            **kwargs: Additional keyword argument to pass to the embed method.

        Yields:
            Iterable[np.ndarray]: The embeddings.
        """
        raise NotImplementedError()
