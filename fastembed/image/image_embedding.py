from typing import Any, Iterable, Optional, Sequence, Type, Union
from dataclasses import asdict

from fastembed.common.types import NumpyArray
from fastembed.common import ImageInput, OnnxProvider
from fastembed.image.image_embedding_base import ImageEmbeddingBase
from fastembed.image.onnx_embedding import OnnxImageEmbedding
from fastembed.common.model_description import DenseModelDescription


class ImageEmbedding(ImageEmbeddingBase):
    EMBEDDINGS_REGISTRY: list[Type[ImageEmbeddingBase]] = [OnnxImageEmbedding]

    @classmethod
    def list_supported_models(cls) -> list[dict[str, Any]]:
        """
        Lists the supported models.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing the model information.

            Example:
                ```
                [
                    {
                        "model": "Qdrant/clip-ViT-B-32-vision",
                        "dim": 512,
                        "description": "CLIP vision encoder based on ViT-B/32",
                        "license": "mit",
                        "size_in_GB": 0.33,
                        "sources": {
                            "hf": "Qdrant/clip-ViT-B-32-vision",
                        },
                        "model_file": "model.onnx",
                    }
                ]
                ```
        """
        return [asdict(model) for model in cls._list_supported_models()]

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        result: list[DenseModelDescription] = []
        for embedding in cls.EMBEDDINGS_REGISTRY:
            result.extend(embedding._list_supported_models())
        return result

    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence[OnnxProvider]] = None,
        cuda: bool = False,
        device_ids: Optional[list[int]] = None,
        lazy_load: bool = False,
        **kwargs: Any,
    ):
        super().__init__(model_name, cache_dir, threads, **kwargs)
        for EMBEDDING_MODEL_TYPE in self.EMBEDDINGS_REGISTRY:
            supported_models = EMBEDDING_MODEL_TYPE._list_supported_models()
            if any(model_name.lower() == model.model.lower() for model in supported_models):
                self.model = EMBEDDING_MODEL_TYPE(
                    model_name,
                    cache_dir,
                    threads=threads,
                    providers=providers,
                    cuda=cuda,
                    device_ids=device_ids,
                    lazy_load=lazy_load,
                    **kwargs,
                )
                return

        raise ValueError(
            f"Model {model_name} is not supported in ImageEmbedding."
            "Please check the supported models using `ImageEmbedding.list_supported_models()`"
        )

    @property
    def embedding_size(self) -> int:
        """Get the embedding size of the current model"""
        if self._embedding_size is None:
            self._embedding_size = self.get_embedding_size(self.model_name)
        return self._embedding_size

    @classmethod
    def get_embedding_size(cls, model_name: str) -> int:
        """Get the embedding size of the passed model

        Args:
            model_name (str): The name of the model to get embedding size for.

        Returns:
            int: The size of the embedding.

        Raises:
            ValueError: If the model name is not found in the supported models.
        """
        descriptions = cls._list_supported_models()
        embedding_size: Optional[int] = None
        for description in descriptions:
            if description.model.lower() == model_name.lower():
                embedding_size = description.dim
                break
        if embedding_size is None:
            model_names = [description.model for description in descriptions]
            raise ValueError(
                f"Embedding size for model {model_name} was None. "
                f"Available model names: {model_names}"
            )
        return embedding_size

    def embed(
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
        yield from self.model.embed(images, batch_size, parallel, **kwargs)
