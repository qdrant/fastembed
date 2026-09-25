import os
import time
import gzip
import json
import shutil
import tarfile
import tempfile
import warnings
import contextlib
from copy import deepcopy
from pathlib import Path, PureWindowsPath
from typing import Any, TypeVar, Generic

import requests
from huggingface_hub import constants, snapshot_download, model_info, list_repo_tree
from huggingface_hub.file_download import repo_folder_name
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import (
    HFValidationError,
    RepositoryNotFoundError,
    disable_progress_bars,
    enable_progress_bars,
)
from loguru import logger
from tqdm import tqdm
from fastembed.common.model_description import BaseModelDescription

T = TypeVar("T", bound=BaseModelDescription)

_DOWNLOAD_CHUNK_SIZE = 256 * 1024


def _hf_transport_errors() -> tuple[type[Exception], ...]:
    """Network errors of huggingface_hub's HTTP library that aren't OSError.

    A refused connection, a DNS failure or a timeout raises an OSError in huggingface_hub 0.x,
    which is built on requests, but a TransportError in 1.x (httpx) and 2.x (httpx2).
    """
    try:
        # huggingface_hub>=1.30 re-exports whichever of httpx and httpx2 it's built on.
        from huggingface_hub.utils import httpx
    except ImportError:
        try:
            import httpx  # huggingface_hub 1.0 to 1.29
        except ImportError:  # huggingface_hub 0.x
            return ()
    return (httpx.TransportError,)


# Errors from an HF download that download_model handles by falling back to url and retrying.
_HF_DOWNLOAD_ERRORS = (OSError, RepositoryNotFoundError, ValueError) + _hf_transport_errors()


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

        response = requests.get(url, stream=True, timeout=(10, 120))

        # Handle HTTP errors
        if response.status_code == 403:
            raise PermissionError(
                "Authentication Error: You do not have permission to access this resource. "
                "Please check your credentials."
            )
        # Otherwise an error page gets written out as though it were the archive.
        response.raise_for_status()

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
                for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                    if chunk:  # Filter out keep-alive new chunks
                        progress_bar.update(len(chunk))
                        file.write(chunk)
        return output_path

    @classmethod
    def _find_legacy_cased_source(cls, cache_dir: str, hf_source_repo: str) -> str | None:
        """Looks for a cached snapshot of the same repo spelled with a different casing.

        Built-in sources used to be lowercase and were canonicalized once it turned out that
        relying on the hub's normalizing redirect breaks proxies. Both the hub and fastembed
        derive the cache directory from the source verbatim, so on a case-sensitive filesystem
        an offline load would otherwise miss a model an older version had already cached.

        Args:
            cache_dir (str): The path to the cache directory.
            hf_source_repo (str): Name of the model on HuggingFace Hub.

        Returns:
            str | None: The differently cased source found in the cache, None if there is none.
        """
        separator = constants.REPO_ID_SEPARATOR
        try:
            expected = repo_folder_name(repo_id=hf_source_repo, repo_type="model")
            entries = list(Path(cache_dir).iterdir())
        except (HFValidationError, OSError):
            return None

        # the repo id sits at the tail of the folder name, with every "/" replaced
        offset = len(expected) - len(hf_source_repo.replace("/", separator))

        for entry in entries:
            if entry.name == expected or entry.name.lower() != expected.lower():
                continue
            if not entry.is_dir():
                continue
            # the hub forbids the separator inside a repo id, so it only marks the split
            return entry.name[offset:].replace(separator, "/")
        return None

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
            hf_source_repo (str): Name of the model on HuggingFace Hub, e.g. "Qdrant/all-MiniLM-L6-v2-onnx".
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
        ) -> dict[str, dict[str, int | str]]:
            meta: dict[str, dict[str, int | str]] = {}
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

        def _save_file_metadata(model_dir: Path, meta: dict[str, dict[str, int | str]]) -> None:
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

        snapshot_dir = Path(cache_dir) / repo_folder_name(
            repo_id=hf_source_repo, repo_type="model"
        )
        metadata_file = snapshot_dir / cls.METADATA_FILE

        if local_files_only:
            disable_progress_bars()
            # a canonical directory can exist yet hold no usable snapshot, e.g. when an
            # earlier download was interrupted, so fall back on failure rather than on
            # the directory being absent
            sources = [hf_source_repo]
            legacy_source = cls._find_legacy_cased_source(cache_dir, hf_source_repo)
            if legacy_source is not None:
                sources.append(legacy_source)

            for index, source in enumerate(sources):
                snapshot_dir = Path(cache_dir) / repo_folder_name(
                    repo_id=source, repo_type="model"
                )
                metadata_file = snapshot_dir / cls.METADATA_FILE
                if metadata_file.exists():
                    metadata = json.loads(metadata_file.read_text())
                    verified = _verify_files_from_metadata(snapshot_dir, metadata, repo_files=[])
                    if not verified:
                        logger.warning(
                            "Local file sizes do not match the metadata."
                        )  # do not raise, still make an attempt to load the model
                try:
                    # a legacy source is only ever resolved against the cache: sending it
                    # to the hub would ask for the very redirect this casing avoids
                    return snapshot_download(
                        repo_id=source,
                        allow_patterns=allow_patterns,
                        cache_dir=cache_dir,
                        local_files_only=True,
                        **kwargs,
                    )
                except _HF_DOWNLOAD_ERRORS:
                    if index == len(sources) - 1:
                        raise
                    logger.info(
                        f"{source} is not usable in {cache_dir}, loading "
                        f"{sources[index + 1]}, cached from the same repo by an older "
                        "fastembed version."
                    )

        # hub sends this request with no timeout unless given one, so an endpoint that accepts
        # the connection but never answers would block here for good, before download_model can
        # retry or fall back to another source. list_repo_tree and snapshot_download's own
        # metadata requests can't be given one, but a silent endpoint now fails here first.
        repo_revision = model_info(hf_source_repo, timeout=constants.HF_HUB_ETAG_TIMEOUT).sha
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

        Nothing is deleted on failure, since `cache_dir` may hold more than this archive.
        Cleaning up a partial extraction is the caller's job.

        Args:
            targz_path (str): Path to the .tar.gz file.
            cache_dir (str): Path to the cache directory.

        Returns:
            cache_dir (str): Path to the cache directory.

        Raises:
            ValueError: If the archive is missing, corrupt, or holds an unsafe member.
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
                if hasattr(tarfile, "data_filter"):
                    tar.extractall(path=cache_dir, filter="data")
                else:
                    # No PEP 706 filter before 3.10.12, so vet the members by hand.
                    members = tar.getmembers()
                    for member in members:
                        cls._validate_tar_member(member)
                    tar.extractall(path=cache_dir, members=members)
                # tarfile stops at the end-of-archive marker, short of the gzip trailer, so
                # the CRC is only checked if the rest of the stream is read.
                while tar.fileobj.read(1 << 20):
                    pass
        except (tarfile.TarError, ValueError, EOFError, gzip.BadGzipFile) as e:
            # gzip raises EOFError for a truncated stream and BadGzipFile for a corrupted one.
            raise ValueError(f"An error occurred while decompressing {targz_path}: {e}") from e

        return cache_dir

    @staticmethod
    def _is_unsafe_tar_path(path: str) -> bool:
        """Checks whether a tar member name or link target may escape the extraction dir.

        Lexical on purpose: resolving against the extraction directory is unsound before
        extraction, since `link/../escape` only escapes once an earlier member has been
        written as a symlink. Any `..` component is therefore rejected outright.
        """
        # PureWindowsPath splits on both separators, so `root` covers POSIX "/evil" as
        # well as "\\evil", which escapes on Windows without being absolute.
        windows_path = PureWindowsPath(path)
        return bool(windows_path.drive or windows_path.root) or ".." in windows_path.parts

    @classmethod
    def _validate_tar_member(cls, member: tarfile.TarInfo) -> None:
        """Raises ValueError if a member could write outside the extraction directory."""
        if cls._is_unsafe_tar_path(member.name):
            raise ValueError(f"Unsafe tar member path: {member.name}")

        if member.issym() or member.islnk():
            if cls._is_unsafe_tar_path(member.linkname):
                raise ValueError(f"Unsafe tar link target: {member.name} -> {member.linkname}")
        elif not (member.isfile() or member.isdir()):
            # Devices, fifos and the like have no place in a model archive.
            raise ValueError(f"Unsupported tar member type: {member.name}")

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
        model_dir = Path(cache_dir) / fast_model_name

        # check if the model_dir and the model files are both present for macOS
        if model_dir.exists() and len(list(model_dir.glob("*"))) > 0:
            if deprecated_tar_struct and not source_url:
                # No built-in model is served from the bucket anymore, only copies of it remain.
                # A model with a url of its own (a custom one) got its copy from there instead.
                # stacklevel points at the caller of TextEmbedding(...) via download_model.
                warnings.warn(
                    f"Loading {model_name} from {model_dir}, a copy downloaded from Google Cloud "
                    "Storage by an older fastembed version. Support for such copies is deprecated "
                    "and will be removed in a future release. To switch to Hugging Face, load the "
                    "model once while huggingface.co (or an `HF_ENDPOINT` mirror) is reachable, "
                    "without `local_files_only=True` or `HF_HUB_OFFLINE=1`, then delete "
                    f"{model_dir}.",
                    FutureWarning,
                    stacklevel=5,
                )
            return model_dir

        if local_files_only:
            logger.error(
                f"Could not find the model tar.gz file at {model_dir} and local_files_only=True."
            )
            raise ValueError(
                f"Could not find the model tar.gz file at {model_dir} and local_files_only=True."
            )

        if cache_tmp_dir.is_symlink():
            raise ValueError(
                f"{cache_tmp_dir} is a symlink, refusing to stage downloads through it"
            )
        cache_tmp_dir.mkdir(parents=True, exist_ok=True)

        # The archive and everything extracted from it go in a directory of this attempt's own,
        # so removing it undoes the attempt without touching any other download of the model.
        staging_dir = Path(tempfile.mkdtemp(dir=cache_tmp_dir, prefix=f"{fast_model_name}-"))
        try:
            model_tar_gz = staging_dir / f"{fast_model_name}.tar.gz"
            cls.download_file_from_gcs(
                source_url,
                output_path=str(model_tar_gz),
            )

            cls.decompress_to_cache(targz_path=str(model_tar_gz), cache_dir=str(staging_dir))

            model_tmp_dir = staging_dir / fast_model_name
            if not model_tmp_dir.is_dir() or model_tmp_dir.is_symlink():
                raise ValueError(
                    f"The archive from {source_url} has no {fast_model_name} directory"
                )

            # Replace a stale empty model_dir, which Windows will not rename onto. rmdir leaves
            # anything else alone, including one another download has just filled.
            with contextlib.suppress(OSError):
                model_dir.rmdir()

            try:
                # Rename from the staging dir to the final name is atomic
                model_tmp_dir.rename(model_dir)
            except OSError:
                # Another download of the same model finished first, so keep its copy.
                if not (model_dir.is_dir() and any(model_dir.iterdir())):
                    raise
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

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
                        "hf": "Qdrant/bge-base-en-v1.5-onnx-Q",
                    }
                }
                ```
            cache_dir (str): The path to the cache directory.
            retries: (int): The number of times to retry (including the first attempt)

        Returns:
            Path: The path to the downloaded model directory.
        """
        local_files_only = kwargs.get("local_files_only", False)
        hf_offline = os.environ.get("HF_HUB_OFFLINE", "").strip().upper()
        if not local_files_only and hf_offline in {"1", "TRUE", "YES", "ON"}:
            local_files_only = True
            kwargs["local_files_only"] = True
        specific_model_path: str | None = kwargs.pop("specific_model_path", None)
        if specific_model_path:
            return Path(specific_model_path)
        retries = 1 if local_files_only else retries
        hf_source = model.sources.hf
        url_source = model.sources.url

        extra_patterns = [model.model_file]
        extra_patterns.extend(model.additional_files)

        if hf_source:
            try:
                cache_kwargs = deepcopy(kwargs)
                cache_kwargs["local_files_only"] = True
                resolved_path = Path(
                    cls.download_files_from_huggingface(
                        hf_source,
                        cache_dir=cache_dir,
                        extra_patterns=extra_patterns,
                        **cache_kwargs,
                    )
                )
                if (resolved_path / model.model_file).exists() and all(
                    (resolved_path / file).exists() for file in extra_patterns
                ):
                    return resolved_path
            except Exception:
                pass
            finally:
                enable_progress_bars()

        sleep = 3.0
        while retries > 0:
            retries -= 1

            if hf_source and not local_files_only:
                # we have already tried loading with `local_files_only=True` via hf and we failed
                try:
                    return Path(
                        cls.download_files_from_huggingface(
                            hf_source,
                            cache_dir=cache_dir,
                            extra_patterns=extra_patterns,
                            **kwargs,
                        )
                    )
                except _HF_DOWNLOAD_ERRORS as e:
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
                        url_source or "",
                        str(cache_dir),
                        deprecated_tar_struct=model.sources.deprecated_tar_struct,
                        local_files_only=local_files_only,
                    )
                except Exception:
                    if not local_files_only:
                        logger.error(f"Could not download model from url: {url_source}")
            elif model.sources.deprecated_tar_struct:
                # Nothing is downloaded from the bucket anymore, but an old copy may still be cached.
                legacy_dir = Path(cache_dir) / f"fast-{model.model.split('/')[-1]}"
                if legacy_dir.is_dir() and any(legacy_dir.iterdir()):
                    return cls.retrieve_model_gcs(
                        model.model,
                        "",
                        str(cache_dir),
                        deprecated_tar_struct=True,
                        local_files_only=True,
                    )

            if local_files_only:
                logger.error("Could not find model in cache_dir")
                break
            else:
                logger.error(
                    f"Could not download model from either source, sleeping for {sleep} seconds, {retries} retries left."
                )
                time.sleep(sleep)
                sleep *= 3

        raise ValueError(f"Could not load model {model.model} from any source.")
