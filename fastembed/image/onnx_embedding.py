from typing import Any, Dict, Iterable, List, Optional, Sequence, Type

import numpy as np

from fastembed.common import ImageInput, OnnxProvider
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.utils import define_cache_dir, normalize
from fastembed.image.image_embedding_base import ImageEmbeddingBase
from fastembed.image.onnx_image_model import ImageEmbeddingWorker, OnnxImageModel

supported_onnx_models = [
    {
        "model": "Qdrant/clip-ViT-B-32-vision",
        "dim": 512,
        "description": "CLIP vision encoder based on ViT-B/32",
        "size_in_GB": 0.34,
        "sources": {
            "hf": "Qdrant/clip-ViT-B-32-vision",
        },
        "model_file": "model.onnx",
    },
    {
        "model": "Qdrant/resnet50-onnx",
        "dim": 2048,
        "description": "ResNet-50 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.",
        "size_in_GB": 0.1,
        "sources": {
            "hf": "Qdrant/resnet50-onnx",
        },
        "model_file": "model.onnx",
    },
]


class OnnxImageEmbedding(ImageEmbeddingBase, OnnxImageModel[np.ndarray]):
    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence[OnnxProvider]] = None,
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

        model_description = self._get_model_description(model_name)
        self.cache_dir = define_cache_dir(cache_dir)
        model_dir = self.download_model(
            model_description, self.cache_dir, local_files_only=self._local_files_only
        )

        self.load_onnx_model(
            model_dir=model_dir,
            model_file=model_description["model_file"],
            threads=threads,
            providers=providers,
        )

    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        """
        Lists the supported models.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the model information.
        """
        return supported_onnx_models

    def embed(
        self,
        images: ImageInput,
        batch_size: int = 16,
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
            cache_dir=str(self.cache_dir),
            images=images,
            batch_size=batch_size,
            parallel=parallel,
            **kwargs,
        )

    @classmethod
    def _get_worker_class(cls) -> Type["ImageEmbeddingWorker"]:
        return OnnxImageEmbeddingWorker

    def _preprocess_onnx_input(
        self, onnx_input: Dict[str, np.ndarray], **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Preprocess the onnx input.
        """

        return onnx_input

    def _post_process_onnx_output(self, output: OnnxOutputContext) -> Iterable[np.ndarray]:
        return normalize(output.model_output).astype(np.float32)


class OnnxImageEmbeddingWorker(ImageEmbeddingWorker):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs) -> OnnxImageEmbedding:
        return OnnxImageEmbedding(model_name=model_name, cache_dir=cache_dir, threads=1, **kwargs)
