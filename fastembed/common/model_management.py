import os
import time
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any, Optional, Union, TypeVar, Generic

import requests
from huggingface_hub import snapshot_download, model_info, list_repo_tree
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import (
    RepositoryNotFoundError,
    disable_progress_bars,
    enable_progress_bars,
)
from loguru import logger
from tqdm import tqdm
from fastembed.common.model_description import BaseModelDescription

T = TypeVar("T", bound=BaseModelDescription)


class ModelManagement(Generic[T]):
    METADATA_FILE = "files_metadata.json"

    @classmethod
    def list_supported_models(cls) -> list[dict[str, Any]]:
        """Lists the supported models.

        Returns:
            list[T]: A list of dictionaries containing the model information.
        """
        raise NotImplementedError()

    @classmethod
    def add_custom_model(
        cls,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Add a custom model to the existing embedding classes based on the passed model descriptions

        Model description dict should contain the fields same as in one of the model descriptions presented
         in fastembed.common.model_description

         E.g. for BaseModelDescription:
              model: str
              sources: ModelSource
              model_file: str
              description: str
              license: str
              size_in_GB: float
              additional_files: list[str]

        Returns:
            None
        """
        raise NotImplementedError()

    @classmethod
    def _list_supported_models(cls) -> list[T]:
        raise NotImplementedError()

    @classmethod
    def _get_model_description(cls, model_name: str) -> T:
        """
        Gets the model description from the model_name.

        Args:
            model_name (str): The name of the model.

        raises:
            ValueError: If the model_name is not supported.

        Returns:
            T: The model description.
        """
        for model in cls._list_supported_models():
            if model_name.lower() == model.model.lower():
                return model

        raise ValueError(f"Model {model_name} is not supported in {cls.__name__}.")

    @classmethod
    def download_file_from_gcs(cls, url: str, output_path: str, show_progress: bool = True) -> str:
        """
        Downloads a file from Google Cloud Storage.

        Args:
            url (str): The URL to download the file from.
            output_path (str): The path to save the downloaded file to.
            show_progress (bool, optional): Whether to show a progress bar. Defaults to True.

        Returns:
            str: The path to the downloaded file.
        """

        if os.path.exists(output_path):
            return output_path
        response = requests.get(url, stream=True)

        # Handle HTTP errors
        if response.status_code == 403:
            raise PermissionError(
                "Authentication Error: You do not have permission to access this resource. "
                "Please check your credentials."
            )

        # Get the total size of the file
        total_size_in_bytes = int(response.headers.get("content-length", 0))

        # Warn if the total size is zero
        if total_size_in_bytes == 0:
            print(f"Warning: Content-length header is missing or zero in the response from {url}.")

        show_progress = bool(total_size_in_bytes and show_progress)

        with tqdm(
            total=total_size_in_bytes,
            unit="iB",
            unit_scale=True,
            disable=not show_progress,
        ) as progress_bar:
            with open(output_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:  # Filter out keep-alive new chunks
                        progress_bar.update(len(chunk))
                        file.write(chunk)
        return output_path

    @classmethod
    def download_files_from_huggingface(
        cls,
        hf_source_repo: str,
        cache_dir: str,
        extra_patterns: list[str],
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        Downloads a model from HuggingFace Hub.
        Args:
            hf_source_repo (str): Name of the model on HuggingFace Hub, e.g. "qdrant/all-MiniLM-L6-v2-onnx".
            cache_dir (Optional[str]): The path to the cache directory.
            extra_patterns (list[str]): extra patterns to allow in the snapshot download, typically
                includes the required model files.
            local_files_only (bool, optional): Whether to only use local files. Defaults to False.
        Returns:
            Path: The path to the model directory.
        """

        def _verify_files_from_metadata(
            model_dir: Path, stored_metadata: dict[str, Any], repo_files: list[RepoFile]
        ) -> bool:
            try:
                for rel_path, meta in stored_metadata.items():
                    file_path = model_dir / rel_path

                    if not file_path.exists():
                        return False

                    if repo_files:  # online verification
                        file_info = next((f for f in repo_files if f.path == file_path.name), None)
                        if (
                            not file_info
                            or file_info.size != meta["size"]
                            or file_info.blob_id != meta["blob_id"]
                        ):
                            return False

                    else:  # offline verification
                        if file_path.stat().st_size != meta["size"]:
                            return False
                return True
            except (OSError, KeyError) as e:
                logger.error(f"Error verifying files: {str(e)}")
                return False

        def _collect_file_metadata(
            model_dir: Path, repo_files: list[RepoFile]
        ) -> dict[str, dict[str, Union[int, str]]]:
            meta: dict[str, dict[str, Union[int, str]]] = {}
            file_info_map = {f.path: f for f in repo_files}
            for file_path in model_dir.rglob("*"):
                if file_path.is_file() and file_path.name != cls.METADATA_FILE:
                    repo_file = file_info_map.get(file_path.name)
                    if repo_file:
                        meta[str(file_path.relative_to(model_dir))] = {
                            "size": repo_file.size,
                            "blob_id": repo_file.blob_id,
                        }
            return meta

        def _save_file_metadata(
            model_dir: Path, meta: dict[str, dict[str, Union[int, str]]]
        ) -> None:
            try:
                if not model_dir.exists():
                    model_dir.mkdir(parents=True, exist_ok=True)
                (model_dir / cls.METADATA_FILE).write_text(json.dumps(meta))
            except (OSError, ValueError) as e:
                logger.warning(f"Error saving metadata: {str(e)}")

        allow_patterns = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "preprocessor_config.json",
        ]

        allow_patterns.extend(extra_patterns)

        snapshot_dir = Path(cache_dir) / f"models--{hf_source_repo.replace('/', '--')}"
        metadata_file = snapshot_dir / cls.METADATA_FILE

        if local_files_only:
            disable_progress_bars()
            if metadata_file.exists():
                metadata = json.loads(metadata_file.read_text())
                verified = _verify_files_from_metadata(snapshot_dir, metadata, repo_files=[])
                if not verified:
                    logger.warning(
                        "Local file sizes do not match the metadata."
                    )  # do not raise, still make an attempt to load the model
            else:
                logger.warning(
                    "Metadata file not found. Proceeding without checking local files."
                )  # if users have downloaded models from hf manually, or they're updating from previous versions of
                # fastembed
            result = snapshot_download(
                repo_id=hf_source_repo,
                allow_patterns=allow_patterns,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                **kwargs,
            )
            return result

        repo_revision = model_info(hf_source_repo).sha
        repo_tree = list(list_repo_tree(hf_source_repo, revision=repo_revision, repo_type="model"))

        allowed_extensions = {".json", ".onnx", ".txt"}
        repo_files = (
            [
                f
                for f in repo_tree
                if isinstance(f, RepoFile) and Path(f.path).suffix in allowed_extensions
            ]
            if repo_tree
            else []
        )

        verified_metadata = False

        if snapshot_dir.exists() and metadata_file.exists():
            metadata = json.loads(metadata_file.read_text())
            verified_metadata = _verify_files_from_metadata(snapshot_dir, metadata, repo_files)

        if verified_metadata:
            disable_progress_bars()

        result = snapshot_download(
            repo_id=hf_source_repo,
            allow_patterns=allow_patterns,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            **kwargs,
        )

        if (
            not verified_metadata
        ):  # metadata is not up-to-date, update it and check whether the files have been
            # downloaded correctly
            metadata = _collect_file_metadata(snapshot_dir, repo_files)

            download_successful = _verify_files_from_metadata(
                snapshot_dir, metadata, repo_files=[]
            )  # offline verification
            if not download_successful:
                raise ValueError(
                    "Files have been corrupted during downloading process. "
                    "Please check your internet connection and try again."
                )
            _save_file_metadata(snapshot_dir, metadata)

        return result

    @classmethod
    def decompress_to_cache(cls, targz_path: str, cache_dir: str) -> str:
        """
        Decompresses a .tar.gz file to a cache directory.

        Args:
            targz_path (str): Path to the .tar.gz file.
            cache_dir (str): Path to the cache directory.

        Returns:
            cache_dir (str): Path to the cache directory.
        """
        # Check if targz_path exists and is a file
        if not os.path.isfile(targz_path):
            raise ValueError(f"{targz_path} does not exist or is not a file.")

        # Check if targz_path is a .tar.gz file
        if not targz_path.endswith(".tar.gz"):
            raise ValueError(f"{targz_path} is not a .tar.gz file.")

        try:
            # Open the tar.gz file
            with tarfile.open(targz_path, "r:gz") as tar:
                # Extract all files into the cache directory
                tar.extractall(
                    path=cache_dir,
                )
        except tarfile.TarError as e:
            # If any error occurs while opening or extracting the tar.gz file,
            # delete the cache directory (if it was created in this function)
            # and raise the error again
            if "tmp" in cache_dir:
                shutil.rmtree(cache_dir)
            raise ValueError(f"An error occurred while decompressing {targz_path}: {e}")

        return cache_dir

    @classmethod
    def retrieve_model_gcs(
        cls,
        model_name: str,
        source_url: str,
        cache_dir: str,
        deprecated_tar_struct: bool = False,
        local_files_only: bool = False,
    ) -> Path:
        fast_model_name = f"{'fast-' if deprecated_tar_struct else ''}{model_name.split('/')[-1]}"
        cache_tmp_dir = Path(cache_dir) / "tmp"
        model_tmp_dir = cache_tmp_dir / fast_model_name
        model_dir = Path(cache_dir) / fast_model_name

        # check if the model_dir and the model files are both present for macOS
        if model_dir.exists() and len(list(model_dir.glob("*"))) > 0:
            return model_dir

        if model_tmp_dir.exists():
            shutil.rmtree(model_tmp_dir)

        cache_tmp_dir.mkdir(parents=True, exist_ok=True)

        model_tar_gz = Path(cache_dir) / f"{fast_model_name}.tar.gz"

        if model_tar_gz.exists():
            model_tar_gz.unlink()

        if not local_files_only:
            cls.download_file_from_gcs(
                source_url,
                output_path=str(model_tar_gz),
            )

            cls.decompress_to_cache(targz_path=str(model_tar_gz), cache_dir=str(cache_tmp_dir))
            assert model_tmp_dir.exists(), f"Could not find {model_tmp_dir} in {cache_tmp_dir}"

            model_tar_gz.unlink()
            # Rename from tmp to final name is atomic
            model_tmp_dir.rename(model_dir)
        else:
            logger.error(
                f"Could not find the model tar.gz file at {model_dir} and local_files_only=True."
            )
            raise ValueError(
                f"Could not find the model tar.gz file at {model_dir} and local_files_only=True."
            )

        return model_dir

    @classmethod
    def download_model(cls, model: T, cache_dir: str, retries: int = 3, **kwargs: Any) -> Path:
        """
        Downloads a model from HuggingFace Hub or Google Cloud Storage.

        Args:
            model (T): The model description.
                Example:
                ```
                {
                    "model": "BAAI/bge-base-en-v1.5",
                    "dim": 768,
                    "description": "Base English model, v1.5",
                    "size_in_GB": 0.44,
                    "sources": {
                        "url": "https://storage.googleapis.com/qdrant-fastembed/fast-bge-base-en-v1.5.tar.gz",
                        "hf": "qdrant/bge-base-en-v1.5-onnx-q",
                    }
                }
                ```
            cache_dir (str): The path to the cache directory.
            retries: (int): The number of times to retry (including the first attempt)

        Returns:
            Path: The path to the downloaded model directory.
        """
        local_files_only = kwargs.get("local_files_only", False)
        specific_model_path: Optional[str] = kwargs.pop("specific_model_path", None)
        if specific_model_path:
            return Path(specific_model_path)
        retries = 1 if local_files_only else retries
        hf_source = model.sources.hf
        url_source = model.sources.url

        sleep = 3.0
        while retries > 0:
            retries -= 1

            if hf_source:
                extra_patterns = [model.model_file]
                extra_patterns.extend(model.additional_files)

                try:
                    return Path(
                        cls.download_files_from_huggingface(
                            hf_source,
                            cache_dir=cache_dir,
                            extra_patterns=extra_patterns,
                            **kwargs,
                        )
                    )
                except (EnvironmentError, RepositoryNotFoundError, ValueError) as e:
                    if not local_files_only:
                        logger.error(
                            f"Could not download model from HuggingFace: {e} "
                            "Falling back to other sources."
                        )
                finally:
                    enable_progress_bars()
            if url_source or local_files_only:
                try:
                    return cls.retrieve_model_gcs(
                        model.model,
                        str(url_source),
                        str(cache_dir),
                        deprecated_tar_struct=model.sources.deprecated_tar_struct,
                        local_files_only=local_files_only,
                    )
                except Exception:
                    if not local_files_only:
                        logger.error(f"Could not download model from url: {url_source}")

            if local_files_only:
                logger.error("Could not find model in cache_dir")
            else:
                logger.error(
                    f"Could not download model from either source, sleeping for {sleep} seconds, {retries} retries left."
                )
            time.sleep(sleep)
            sleep *= 3

        raise ValueError(f"Could not load model {model.model} from any source.")
