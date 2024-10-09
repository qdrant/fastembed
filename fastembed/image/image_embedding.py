from typing import Any, Dict, Iterable, List, Optional, Sequence, Type

import numpy as np

from fastembed.common import ImageInput, OnnxProvider
from fastembed.image.image_embedding_base import ImageEmbeddingBase
from fastembed.image.onnx_embedding import OnnxImageEmbedding


class ImageEmbedding(ImageEmbeddingBase):
    EMBEDDINGS_REGISTRY: List[Type[ImageEmbeddingBase]] = [OnnxImageEmbedding]

    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        """
        Lists the supported models.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the model information.

            Example:
                ```
                [
                    {
                        "model": "Qdrant/clip-ViT-B-32-vision",
                        "dim": 512,
                        "description": "CLIP vision encoder based on ViT-B/32",
                        "size_in_GB": 0.33,
                        "sources": {
                            "hf": "Qdrant/clip-ViT-B-32-vision",
                        },
                        "model_file": "model.onnx",
                    }
                ]
                ```
        """
        result = []
        for embedding in cls.EMBEDDINGS_REGISTRY:
            result.extend(embedding.list_supported_models())
        return result

    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence[OnnxProvider]] = None,
        cuda: bool = False,
        device_ids: Optional[List[int]] = None,
        lazy_load: bool = False,
        **kwargs,
    ):
        super().__init__(model_name, cache_dir, threads, **kwargs)
        self.lazy_load = lazy_load
        self.providers = providers
        self.device_ids = device_ids
        self.model = None
        self.model_class = None
        self.kwargs = kwargs

        self.cuda = cuda or any(
            (
                p == "CUDAExecutionProvider"
                if isinstance(p, str)
                else p[0] == "CUDAExecutionProvider"
            )
            for p in providers
        )

        for EMBEDDING_MODEL_TYPE in self.EMBEDDINGS_REGISTRY:
            supported_models = EMBEDDING_MODEL_TYPE.list_supported_models()
            if any(model_name.lower() == model["model"].lower() for model in supported_models):
                self.model_class = EMBEDDING_MODEL_TYPE
                if not self.lazy_load:
                    self._load_onnx_model()
                return

        raise ValueError(
            f"Model {model_name} is not supported in ImageEmbedding."
            "Please check the supported models using `ImageEmbedding.list_supported_models()`"
        )

    def _load_onnx_model(self):
        self.model = self.model_class(
            self.model_name,
            self.cache_dir,
            threads=self.threads,
            providers=self.providers,
            cuda=self.cuda,
            **self.kwargs,
        )

    def embed(
        self,
        images: ImageInput,
        batch_size: int = 16,
        parallel: Optional[int] = None,
        **kwargs,
    ) -> Iterable[np.ndarray]:
        """
        Encode a list of documents into list of embeddings.
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
        if self.lazy_load and self.model is None and parallel is None:
            self._load_onnx_model()

        if parallel:
            yield from self.model_class._embed_images_parallel(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
                images=images,
                batch_size=batch_size,
                parallel=parallel,
                providers=self.providers,
                cuda=self.cuda,
                device_ids=self.device_ids,
                **{**self.kwargs, **kwargs},
            )
        else:
            yield from self.model.embed(images, batch_size, parallel, **kwargs)
