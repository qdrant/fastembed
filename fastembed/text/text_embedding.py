from typing import Any, Dict, Iterable, List, Optional, Sequence, Type, Union

import numpy as np

from fastembed.common import OnnxProvider
from fastembed.text.clip_embedding import CLIPOnnxEmbedding
from fastembed.text.e5_onnx_embedding import E5OnnxEmbedding
from fastembed.text.pooled_normalized_embedding import PooledNormalizedEmbedding
from fastembed.text.pooled_embedding import PooledEmbedding
from fastembed.text.onnx_embedding import OnnxTextEmbedding
from fastembed.text.text_embedding_base import TextEmbeddingBase


class TextEmbedding(TextEmbeddingBase):
    EMBEDDINGS_REGISTRY: List[Type[TextEmbeddingBase]] = [
        OnnxTextEmbedding,
        E5OnnxEmbedding,
        CLIPOnnxEmbedding,
        PooledNormalizedEmbedding,
        PooledEmbedding,
    ]

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
                        "model": "intfloat/multilingual-e5-large",
                        "dim": 1024,
                        "description": "Multilingual model, e5-large. Recommend using this model for non-English languages",
                        "size_in_GB": 2.24,
                        "sources": {
                            "gcp": "https://storage.googleapis.com/qdrant-fastembed/fast-multilingual-e5-large.tar.gz",
                            "hf": "qdrant/multilingual-e5-large-onnx",
                        }
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
        model_name: str = "BAAI/bge-small-en-v1.5",
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
            f"Model {model_name} is not supported in TextEmbedding."
            "Please check the supported models using `TextEmbedding.list_supported_models()`"
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
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs,
    ) -> Iterable[np.ndarray]:
        """
        Encode a list of documents into list of embeddings.
        We use mean pooling with attention so that the model can handle variable-length inputs.

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
        if self.lazy_load and self.model is None and parallel is None:
            self._load_onnx_model()

        if parallel:
            yield from self.model_class._embed_documents_parallel(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
                documents=documents,
                batch_size=batch_size,
                parallel=parallel,
                providers=self.providers,
                cuda=self.cuda,
                device_ids=self.device_ids,
                **{**self.kwargs, **kwargs},
            )
        else:
            yield from self.model.embed(documents, batch_size, parallel, **kwargs)
