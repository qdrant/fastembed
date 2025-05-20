from typing import Any, Iterable, Optional, Sequence, Type, Union


from fastembed.common.types import NumpyArray
from fastembed.common import ImageInput, OnnxProvider
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.utils import define_cache_dir, normalize
from fastembed.image.image_embedding_base import ImageEmbeddingBase
from fastembed.image.onnx_image_model import ImageEmbeddingWorker, OnnxImageModel

from fastembed.common.model_description import DenseModelDescription, ModelSource

supported_onnx_models: list[DenseModelDescription] = [
    DenseModelDescription(
        model="Qdrant/clip-ViT-B-32-vision",
        dim=512,
        description="Image embeddings, Multimodal (text&image), 2021 year",
        license="mit",
        size_in_GB=0.34,
        sources=ModelSource(hf="Qdrant/clip-ViT-B-32-vision"),
        model_file="model.onnx",
    ),
    DenseModelDescription(
        model="Qdrant/resnet50-onnx",
        dim=2048,
        description="Image embeddings, Unimodal (image), 2016 year",
        license="apache-2.0",
        size_in_GB=0.1,
        sources=ModelSource(hf="Qdrant/resnet50-onnx"),
        model_file="model.onnx",
    ),
    DenseModelDescription(
        model="Qdrant/Unicom-ViT-B-16",
        dim=768,
        description="Image embeddings (more detailed than Unicom-ViT-B-32), Multimodal (text&image), 2023 year",
        license="apache-2.0",
        size_in_GB=0.82,
        sources=ModelSource(hf="Qdrant/Unicom-ViT-B-16"),
        model_file="model.onnx",
    ),
    DenseModelDescription(
        model="Qdrant/Unicom-ViT-B-32",
        dim=512,
        description="Image embeddings, Multimodal (text&image), 2023 year",
        license="apache-2.0",
        size_in_GB=0.48,
        sources=ModelSource(hf="Qdrant/Unicom-ViT-B-32"),
        model_file="model.onnx",
    ),
    DenseModelDescription(
        model="jinaai/jina-clip-v1",
        dim=768,
        description="Image embeddings, Multimodal (text&image), 2024 year",
        license="apache-2.0",
        size_in_GB=0.34,
        sources=ModelSource(hf="jinaai/jina-clip-v1"),
        model_file="onnx/vision_model.onnx",
    ),
]


class OnnxImageEmbedding(ImageEmbeddingBase, OnnxImageModel[NumpyArray]):
    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence[OnnxProvider]] = None,
        cuda: bool = False,
        device_ids: Optional[list[int]] = None,
        lazy_load: bool = False,
        device_id: Optional[int] = None,
        specific_model_path: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initializes an ONNX image embedding model with configurable device, threading, and loading options.
        
        Args:
            model_name: Name of the ONNX model to use, in the format <org>/<model>.
            cache_dir: Optional directory for caching model files.
            threads: Number of threads for ONNX runtime session.
            providers: Optional list of ONNX runtime providers to use for inference.
            cuda: If True, enables CUDA for inference; mutually exclusive with `providers`.
            device_ids: Optional list of device IDs for parallel processing; used with `cuda=True`.
            lazy_load: If True, defers model loading until first use.
            device_id: Optional device ID for model loading in the current process.
            specific_model_path: Optional path to a specific ONNX model directory.
        
        Raises:
            ValueError: If `model_name` is not in the required <org>/<model> format.
        """

        super().__init__(model_name, cache_dir, threads, **kwargs)
        self.providers = providers
        self.lazy_load = lazy_load

        # List of device ids, that can be used for data parallel processing in workers
        self.device_ids = device_ids
        self.cuda = cuda

        # This device_id will be used if we need to load model in current process
        self.device_id: Optional[int] = None
        if device_id is not None:
            self.device_id = device_id
        elif self.device_ids is not None:
            self.device_id = self.device_ids[0]

        self.model_description = self._get_model_description(model_name)
        self.cache_dir = str(define_cache_dir(cache_dir))
        self._specific_model_path = specific_model_path
        self._model_dir = self.download_model(
            self.model_description,
            self.cache_dir,
            local_files_only=self._local_files_only,
            specific_model_path=self._specific_model_path,
        )

        if not self.lazy_load:
            self.load_onnx_model()

    def load_onnx_model(self) -> None:
        """
        Load the onnx model.
        """
        self._load_onnx_model(
            model_dir=self._model_dir,
            model_file=self.model_description.model_file,
            threads=self.threads,
            providers=self.providers,
            cuda=self.cuda,
            device_id=self.device_id,
        )

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        """
        Lists the supported models.

        Returns:
            list[DenseModelDescription]: A list of DenseModelDescription objects containing the model information.
        """
        return supported_onnx_models

    def embed(
        self,
        images: Union[ImageInput, Iterable[ImageInput]],
        batch_size: int = 16,
        parallel: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        """
        Generates embeddings for one or more images using the loaded ONNX model.
        
        Args:
            images: A single image input or an iterable of image inputs to embed.
            batch_size: Number of images to process in each batch.
            parallel: Number of parallel workers to use for data-parallel encoding. If 0, uses all available cores; if None, disables parallel processing.
        
        Returns:
            An iterable of numpy arrays, each representing the embedding of an input image.
        """

        yield from self._embed_images(
            model_name=self.model_name,
            cache_dir=str(self.cache_dir),
            images=images,
            batch_size=batch_size,
            parallel=parallel,
            providers=self.providers,
            cuda=self.cuda,
            device_ids=self.device_ids,
            local_files_only=self._local_files_only,
            specific_model_path=self._specific_model_path,
            **kwargs,
        )

    @classmethod
    def _get_worker_class(cls) -> Type["ImageEmbeddingWorker[NumpyArray]"]:
        return OnnxImageEmbeddingWorker

    def _preprocess_onnx_input(
        self, onnx_input: dict[str, NumpyArray], **kwargs: Any
    ) -> dict[str, NumpyArray]:
        """
        Preprocess the onnx input.
        """

        return onnx_input

    def _post_process_onnx_output(
        self, output: OnnxOutputContext, **kwargs: Any
    ) -> Iterable[NumpyArray]:
        return normalize(output.model_output)


class OnnxImageEmbeddingWorker(ImageEmbeddingWorker[NumpyArray]):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs: Any) -> OnnxImageEmbedding:
        return OnnxImageEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
