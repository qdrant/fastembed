import os
import shutil
import tarfile
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List

import numpy as np
import onnxruntime as ort
import requests
from tokenizers import Tokenizer
from tqdm import tqdm

# set the default logger to ERROR level with this
ort.set_default_logger_severity(3)


# Use pytorches default epsilon for division by zero
# https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
def normalize(v):
    norm = np.linalg.norm(v, axis=1)
    norm[norm == 0] = 1e-12
    return v / norm[:, np.newaxis]


class Embedding(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> List[np.ndarray]:
        pass


class ONNXProviders:
    """List of Execution Providers: https://onnxruntime.ai/docs/execution-providers"""
    CPU = "CPUExecutionProvider"
    GPU = "CUDAExecutionProvider"
    Metal = "CoreMLExecutionProvider"

class DefaultEmbedding(Embedding):
    def __init__(self, 
                 model_name: str = "BAAI/bge-base-en", 
                 onnx_providers: List[str] = [ONNXProviders.Metal], 
                 max_length: int = 512):
        self.cache_dir = Path(tempfile.gettempdir()) / "fastembed"
        assert "/" in model_name, "model_name must be in the format <org>/<model> e.g. BAAI/bge-base-en"
        model_name = model_name.split("/")[-1]
        fast_model_name = f"fast-{model_name}"
        filepath = self.download_file_from_gcs(
            f"https://storage.googleapis.com/qdrant-fastembed/{fast_model_name}.tar.gz",
            output_path=f"{fast_model_name}.tar.gz",
        )
        model_dir = self.decompress_to_cache(targz_path=filepath)
        model_dir = Path(model_dir) / fast_model_name
        tokenizer_path = model_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            raise ValueError(f"Could not find tokenizer.json in {model_dir}")
        model_path = model_dir / "model_optimized.onnx"
        if not model_path.exists():
            raise ValueError(f"Could not find model_optimized.onnx in {model_dir}")

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=max_length)
        self.model = ort.InferenceSession(str(model_path), providers=onnx_providers)

    def download_file_from_gcs(self, url: str, output_path: str, show_progress: bool = True):
        if os.path.exists(output_path):
            return output_path
        response = requests.get(url, stream=True)

        # Handle HTTP errors
        if response.status_code == 403:
            print("Authentication error: you do not have permission to access this resource.")
            return
        elif response.status_code != 200:
            print(f"HTTP error {response.status_code} while trying to download the file.")
            return

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

    def decompress_to_cache(self, targz_path: str, cache_dir: str = None):
        # Check if targz_path exists and is a file
        if not os.path.isfile(targz_path):
            raise ValueError(f"{targz_path} does not exist or is not a file.")

        # Check if targz_path is a .tar.gz file
        if not targz_path.endswith(".tar.gz"):
            raise ValueError(f"{targz_path} is not a .tar.gz file.")

        # Create a temporary directory for caching if cache_dir is not provided
        if cache_dir is None:
            cache_dir = self.cache_dir

        # Decompress the tar.gz file if it has not been decompressed already
        if Path(cache_dir).exists():
            return cache_dir

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

    def encode(self, documents: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Encode a list of documents into list of embeddings.
        We use mean pooling with attention so that the model can handle variable-length inputs.

        Args:
            documents: List of documents to encode
            batch_size: Batch size for encoding -- higher values will use more memory, but be faster
        
        Returns:
            List of embeddings, one per document
        """
        all_embeddings = []

        # TODO: Replace loop with parallelized batching
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            encoded = [self.tokenizer.encode(d) for d in batch]
            input_ids = np.array([e.ids for e in encoded])
            attention_mask = np.array([e.attention_mask for e in encoded])
            onnx_input = {
                "input_ids": np.array(input_ids, dtype=np.int64),
                "attention_mask": np.array(attention_mask, dtype=np.int64),
                "token_type_ids": np.array([np.zeros(len(e), dtype=np.int64) for e in input_ids], dtype=np.int64),
            }
            model_output = self.model.run(None, onnx_input)
            last_hidden_state = model_output[0]
            # Perform mean pooling with attention weighting
            input_mask_expanded = np.broadcast_to(np.expand_dims(attention_mask, -1), last_hidden_state.shape)
            embeddings = np.sum(last_hidden_state * input_mask_expanded, 1) / np.clip(
                input_mask_expanded.sum(1), a_min=1e-9, a_max=None
            )
            embeddings = normalize(embeddings).astype(np.float32)
            all_embeddings.append(embeddings)
        return np.concatenate(all_embeddings)


class SentenceTransformersEmbedding(Embedding):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Please install the sentence-transformers package to use this method.")
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(texts)


class GeneralTextEmbedding(Embedding):
    """
    https://huggingface.co/thenlper/gte-large

    SoTA embedding model for text based retrieval tasks.
    """

    @classmethod
    def average_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def __init__(self, model_name="thenlper/gte-large"):
        try:
            import torch.nn.functional as F
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError("Please install the transformers package with torch to use this method.")
        self.tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
        self.model = AutoModel.from_pretrained("thenlper/gte-large")

    def encode(self, input_texts: List[str]):
        try:
            import torch.nn.functional as F
        except ImportError:
            raise ImportError("Please install Pytorch to use this method.")
        # Tokenize the input texts
        batch_dict = self.tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors="pt")

        outputs = self.model(**batch_dict)
        embeddings = GeneralTextEmbedding.average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])

        # (Optionally) normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        scores = (embeddings[:1] @ embeddings[1:].T) * 100
        return scores


class OpenAIEmbedding(Embedding):
    def __init__(self):
        # Initialize your OpenAI model here
        # self.model = ...
        ...

    def encode(self, texts):
        # Use your OpenAI model to encode the texts
        # return self.model.encode(texts)
        raise NotImplementedError
