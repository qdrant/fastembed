import os
import shutil
import tarfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List

import numpy as np
import requests
from optimum.onnxruntime import ORTModelForFeatureExtraction
from tqdm import tqdm
from transformers import AutoTokenizer


def normalize(input_array, p=2.0, dim=1, eps=1e-12):
    # Calculate the Lp norm along the specified dimension
    norm = np.linalg.norm(input_array, ord=p, axis=dim, keepdims=True)
    norm = np.maximum(norm, eps) # Avoid division by zero
    normalized_array = input_array / norm
    return normalized_array

class Embedding(ABC):
    """
    Abstract class for embeddings.

    Args:
        ABC ():

    Raises:
        NotImplementedError: Raised when you call an abstract method that has not been implemented.
        PermissionError: _description_
        ValueError: Several possible reasons: 1) targz_path does not exist or is not a file, 2) targz_path is not a .tar.gz file, 3) An error occurred while decompressing targz_path, 4) Could not find model_dir in cache_dir, 5) Could not find tokenizer.json in model_dir, 6) Could not find model.onnx in model_dir.
        NotImplementedError: _description_

    Returns:
        _type_: _description_

    Yields:
        _type_: _description_
    """

    @abstractmethod
    def embed(self, texts: List[str]) -> List[np.ndarray]:
        raise NotImplementedError

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
                "Authentication Error: You do not have permission to access this resource. Please check your credentials."
            )

        # Get the total size of the file
        total_size_in_bytes = int(response.headers.get("content-length", 0))

        # Warn if the total size is zero
        if total_size_in_bytes == 0:
            print(f"Warning: Content-length header is missing or zero in the response from {url}.")

        # Initialize the progress bar
        progress_bar = (
            tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
            if total_size_in_bytes and show_progress
            else None
        )

        # Attempt to download the file
        try:
            with open(output_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):  # Adjust chunk size to your preference
                    if chunk:  # Filter out keep-alive new chunks
                        if progress_bar is not None:
                            progress_bar.update(len(chunk))
                        file.write(chunk)
        except Exception as e:
            print(f"An error occurred while trying to download the file: {str(e)}")
            return
        finally:
            if progress_bar is not None:
                progress_bar.close()
        return output_path

    @classmethod
    def decompress_to_cache(cls, targz_path: str, cache_dir: str):
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
                tar.extractall(path=cache_dir)
        except tarfile.TarError as e:
            # If any error occurs while opening or extracting the tar.gz file,
            # delete the cache directory (if it was created in this function)
            # and raise the error again
            if "tmp" in cache_dir:
                shutil.rmtree(cache_dir)
            raise ValueError(f"An error occurred while decompressing {targz_path}: {e}")

        return cache_dir

    def retrieve_model(self, model_name: str, cache_dir: str) -> Path:
        """
        Retrieves a model from Google Cloud Storage.

        Args:
            model_name (str): The name of the model to retrieve.
            cache_dir (str): The path to the cache directory.

        Raises:
            ValueError: If the model_name is not in the format <org>/<model> e.g. BAAI/bge-base-en.

        Returns:
            Path: The path to the model directory.
        """

        assert "/" in model_name, "model_name must be in the format <org>/<model> e.g. BAAI/bge-base-en"

        fast_model_name = f"fast-{model_name.split('/')[-1]}"

        model_dir = Path(cache_dir) / fast_model_name
        if model_dir.exists():
            return model_dir

        model_tar_gz = Path(cache_dir) / f"{fast_model_name}.tar.gz"
        try:
            self.download_file_from_gcs(
                f"https://storage.googleapis.com/qdrant-fastembed/{fast_model_name}.tar.gz",
            output_path=str(model_tar_gz),
            )
        except PermissionError:
            simple_model_name = model_name.replace("/", "-")
            print(f"Was not able to download {fast_model_name}.tar.gz, trying {simple_model_name}.tar.gz")
            self.download_file_from_gcs(
                f"https://storage.googleapis.com/qdrant-fastembed/{simple_model_name}.tar.gz",
                output_path=str(model_tar_gz),
            )

        self.decompress_to_cache(targz_path=str(model_tar_gz), cache_dir=cache_dir)
        assert model_dir.exists(), f"Could not find {model_dir} in {cache_dir}"

        model_tar_gz.unlink()

        return model_dir

    def passage_embed(self, texts: List[str], batch_size: int = 256) -> Iterable[np.ndarray]:
        """
        Embeds a list of text passages into a list of embeddings.

        Args:
            texts (List[str]): The list of texts to embed.
            batch_size (int, optional): The batch size. Defaults to 256.

        Yields:
            Iterable[np.ndarray]: The embeddings.
        """

        for i in range(0, len(texts), batch_size):
            # Prepend "passage: " to each text
            yield from self.embed([f"passage: {t}" for t in texts[i : i + batch_size]])

    def query_embed(self, query: str) -> Iterable[np.ndarray]:
        """
        Embeds a query

        Args:
            query (str): The query to search for.

        Returns:
            Iterable[np.ndarray]: The embeddings.
        """

        # Prepend "query: " to the query
        query = f"query: {query}"
        # Embed the query
        query_embedding = self.embed([query])
        # Compute the cosine similarity between the query embedding and the document embeddings
        return query_embedding


class FlagEmbedding(Embedding):
    """
    Implementation of the Flag Embedding model.

    Args:
        Embedding (_type_): _description_
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        max_length: int = 512,
        cache_dir: str = None,
    ):
        """
        Args:
            model_name (str): The name of the model to use.
            max_length (int, optional): The maximum number of tokens. Defaults to 512. Unknown behavior for values > 512.

        Raises:
            ValueError: If the model_name is not in the format <org>/<model> e.g. BAAI/bge-base-en.
        """
        if cache_dir is None:
            cache_dir = Path(".").resolve() / "local_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
        model_dir = self.retrieve_model(model_name, cache_dir)
        if not (model_dir / "tokenizer.json").exists():
            raise ValueError(f"Could not find tokenizer.json in {model_dir}")
        if not (model_dir / "model.onnx").exists():
            # Rename file model_optimized.onnx to model.onnx if it exists
            if (model_dir / "model_optimized.onnx").exists():
                (model_dir / "model_optimized.onnx").rename(model_dir / "model.onnx")
            else:
                raise ValueError(f"Could not find model.onnx in {model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = ORTModelForFeatureExtraction.from_pretrained(str(model_dir))

    def onnx_embed(self, documents: List[str]) -> Iterable[np.ndarray]:
        encoded_input = self.tokenizer(documents, padding=True, truncation=True, return_tensors='pt')
        model_output = self.model(**encoded_input)
        embeddings = model_output[0][:, 0]
        return normalize(embeddings, p=2, dim=1)

    def embed(self, documents: List[str], batch_size: int = 256) -> Iterable[np.ndarray]:
        """
        Encode a list of documents into list of embeddings.
        We use mean pooling with attention so that the model can handle variable-length inputs.

        Args:
            documents: List of documents to embed
            batch_size: Batch size for encoding -- higher values will use more memory, but be faster

        Returns:
            List of embeddings, one per document
        """
        # TODO: Replace loop with parallelized batching
        if len(documents) >= batch_size:
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                return self.onnx_embed(batch)
        else:
            return self.onnx_embed(documents)


class DefaultEmbedding(FlagEmbedding):
    """
    Implementation of the default Flag Embedding model.

    Args:
        FlagEmbedding (_type_): _description_
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        onnx_providers: List[str] = None,
        max_length: int = 512,
        cache_dir: str = None,
    ):
        # if onnx_providers is None:
        #     onnx_providers = [ONNXProviders.CPU]
        super().__init__(model_name, max_length=max_length, cache_dir=cache_dir)


class OpenAIEmbedding(Embedding):
    def __init__(self):
        # Initialize your OpenAI model here
        # self.model = ...
        ...

    def embed(self, texts):
        # Use your OpenAI model to embed the texts
        # return self.model.embed(texts)
        raise NotImplementedError
        raise NotImplementedError
