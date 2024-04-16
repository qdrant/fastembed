from pathlib import Path
from typing import List, Optional, Dict, Any

from huggingface_hub import snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError
from loguru import logger


def locate_model_file(model_dir: Path, file_names: List[str]) -> Path:
    """
    Find model path for both TransformerJS style `onnx`  subdirectory structure and direct model weights structure used
    by Optimum and Qdrant
    """
    if not model_dir.is_dir():
        raise ValueError(f"Provided model path '{model_dir}' is not a directory.")

    for file_name in file_names:
        file_paths = [path for path in model_dir.rglob(file_name) if path.is_file()]

        if file_paths:
            return file_paths[0]

    raise ValueError(f"Could not find either of {', '.join(file_names)} in {model_dir}")


class ModelManagement:
    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        """Lists the supported models.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the model information.
        """
        raise NotImplementedError()

    @classmethod
    def _get_model_description(cls, model_name: str) -> Dict[str, Any]:
        """
        Gets the model description from the model_name.

        Args:
            model_name (str): The name of the model.

        raises:
            ValueError: If the model_name is not supported.

        Returns:
            Dict[str, Any]: The model description.
        """
        for model in cls.list_supported_models():
            if model_name.lower() == model["model"].lower():
                return model

        raise ValueError(f"Model {model_name} is not supported in {cls.__name__}.")

    @classmethod
    def download_files_from_huggingface(
        cls, hf_source_repo: str, cache_dir: Optional[str] = None
    ) -> str:
        """
        Downloads a model from HuggingFace Hub.
        Args:
            hf_source_repo (str): Name of the model on HuggingFace Hub, e.g. "qdrant/all-MiniLM-L6-v2-onnx".
            cache_dir (Optional[str]): The path to the cache directory.
        Returns:
            Path: The path to the model directory.
        """

        return snapshot_download(
            repo_id=hf_source_repo,
            allow_patterns=[
                "*.onnx",
                "*.onnx_data",
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ],
            cache_dir=cache_dir,
        )

    @classmethod
    def download_model(cls, model: Dict[str, Any], cache_dir: Path) -> Path:
        """
        Downloads a model from HuggingFace Hub or Google Cloud Storage.

        Args:
            model (Dict[str, Any]): The model description.
                Example:
                ```
                {
                    "model": "BAAI/bge-base-en-v1.5",
                    "dim": 768,
                    "description": "Base English model, v1.5",
                    "size_in_GB": 0.44,
                    "sources": {
                        "hf": "qdrant/bge-base-en-v1.5-onnx-q",
                    }
                }
                ```
            cache_dir (str): The path to the cache directory.

        Returns:
            Path: The path to the downloaded model directory.
        """

        hf_source = model.get("sources", {}).get("hf")

        if hf_source:
            try:
                return Path(
                    cls.download_files_from_huggingface(hf_source, cache_dir=str(cache_dir))
                )
            except (EnvironmentError, RepositoryNotFoundError, ValueError) as e:
                logger.error(
                    f"Could not download model from HuggingFace: {e}"
                    "Falling back to other sources."
                )

        raise ValueError(f"Could not download model {model['model']} from any source.")
