from typing import Any, Iterable, Optional, Sequence, Type, Union

import numpy as np

from fastembed.common import OnnxProvider, ImageInput
from fastembed.late_interaction_multimodal.colpali import ColPali

from fastembed.late_interaction_multimodal.late_interaction_multimodal_embedding_base import (
    LateInteractionMultimodalEmbeddingBase,
)


class LateInteractionMultimodalEmbedding(LateInteractionMultimodalEmbeddingBase):
    EMBEDDINGS_REGISTRY: list[Type[LateInteractionMultimodalEmbeddingBase]] = [ColPali]

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
                         "model": "AndrewOgn/colpali-v1.3-merged-onnx",
                         "dim": 128,
                         "description": "Text embeddings, Unimodal (text), Aligned to image latent space, ColBERT-compatible, 512 tokens max, 2024.",
                         "license": "mit",
                         "size_in_GB": 6.06,
                         "sources": {
                            "hf": "AndrewOgn/colpali-v1.3-merged-onnx",
                            },
                         "additional_files": [
                         "model.onnx_data",
                ],
                "model_file": "model.onnx",
                    },
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
        device_ids: Optional[list[int]] = None,
        lazy_load: bool = False,
        **kwargs,
    ):
        super().__init__(model_name, cache_dir, threads, **kwargs)
        for EMBEDDING_MODEL_TYPE in self.EMBEDDINGS_REGISTRY:
            supported_models = EMBEDDING_MODEL_TYPE.list_supported_models()
            if any(model_name.lower() == model["model"].lower() for model in supported_models):
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
            f"Model {model_name} is not supported in LateInteractionMultimodalEmbedding."
            "Please check the supported models using `LateInteractionMultimodalEmbedding.list_supported_models()`"
        )

    def embed_text(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs,
    ) -> Iterable[np.ndarray]:
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
        yield from self.model.embed_text(documents, batch_size, parallel, **kwargs)

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
        yield from self.model.embed_image(images, batch_size, parallel, **kwargs)
