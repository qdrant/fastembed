from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Iterable, Type

import numpy as np

from fastembed.common.onnx_model import EmbeddingWorker
from fastembed.common.models import normalize
from fastembed.common.utils import define_cache_dir
from fastembed.image.image_embedding_base import ImageEmbeddingBase
from fastembed.image.onnx_image_model import OnnxImageModel

supported_onnx_models = [
    {
        "model": "canavar/clip-ViT-B-32-multilingual-v1-ONNX",
        "dim": 512,
        "description": "CLIP",
        "size_in_GB": 0.51,
        "sources": {
            "hf": "canavar/clip-ViT-B-32-multilingual-v1-ONNX",
        },
    }
]


class OnnxImageEmbedding(ImageEmbeddingBase, OnnxImageModel[np.ndarray]):
    def __init__(
        self,
        model_name: str = "...",
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            model_name (str): The name of the model to use.
            cache_dir (str, optional): The path to the cache directory.
                                       Can be set using the `FASTEMBED_CACHE_PATH` env variable.
                                       Defaults to `fastembed_cache` in the system's temp directory.
            threads (int, optional): The number of threads single onnxruntime session can use. Defaults to None.

        Raises:
            ValueError: If the model_name is not in the format <org>/<model> e.g. BAAI/bge-base-en.
        """

        super().__init__(model_name, cache_dir, threads, **kwargs)

        self.model_name = model_name
        self._model_description = self._get_model_description(model_name)

        self._cache_dir = define_cache_dir(cache_dir)
        self._model_dir = self.download_model(self._model_description, self._cache_dir)

        self.load_onnx_model(self._model_dir, self.threads)

    def embed(
        self,
        images: Union[Union[str, Path], Iterable[Union[str, Path]]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs,
    ) -> Iterable[np.ndarray]:
        """
        Encode a list of images into list of embeddings.
        We use mean pooling with attention so that the model can handle variable-length inputs.

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
            cache_dir=str(self._cache_dir),
            images=images,
            batch_size=batch_size,
            parallel=parallel,
        )

    @classmethod
    def _get_worker_class(cls) -> Type["EmbeddingWorker"]:
        return OnnxImageEmbeddingWorker

    def _preprocess_onnx_input(self, onnx_input: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Preprocess the onnx input.
        """
        return onnx_input

    @classmethod
    def _post_process_onnx_output(
        cls, output: Tuple[np.ndarray, np.ndarray]
    ) -> Iterable[np.ndarray]:
        embeddings, _ = output
        return normalize(embeddings[:, 0]).astype(np.float32)


class OnnxImageEmbeddingWorker(EmbeddingWorker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
    ) -> OnnxImageEmbedding:
        return OnnxImageEmbedding(model_name=model_name, cache_dir=cache_dir, threads=1)
