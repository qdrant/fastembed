from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence, Type

import numpy as np

from fastembed.common.model_description import DenseModelDescription, ModelSource
from fastembed.common.model_management import ModelManagement
from fastembed.common.onnx_model import OnnxModel, OnnxOutputContext, EmbeddingWorker
from fastembed.common.types import NumpyArray, OnnxProvider, Device
from fastembed.common.utils import define_cache_dir, iter_batch, normalize


supported_protein_models: list[DenseModelDescription] = [
    DenseModelDescription(
        model="facebook/esm2_t12_35M_UR50D",
        dim=480,
        description="Protein embeddings, ESM-2 35M parameters, 480 dimensions, 1024 max sequence length",
        license="mit",
        size_in_GB=0.13,
        sources=ModelSource(hf="facebook/esm2_t12_35M_UR50D"),
        model_file="model.onnx",
        additional_files=["vocab.txt"],
    ),
]


class ProteinTokenizer:
    """
    Simple tokenizer for protein sequences using ESM-2 vocabulary.
    """

    def __init__(self, vocab_path: Path, max_length: int = 1024):
        self.max_length = max_length
        self.vocab: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

        with open(vocab_path) as f:
            for idx, line in enumerate(f):
                token = line.strip()
                self.vocab[token] = idx
                self.id_to_token[idx] = token

        self.cls_token_id = self.vocab.get("<cls>", 0)
        self.eos_token_id = self.vocab.get("<eos>", 2)
        self.pad_token_id = self.vocab.get("<pad>", 1)
        self.unk_token_id = self.vocab.get("<unk>", 3)

    def encode(self, sequence: str) -> tuple[list[int], list[int]]:
        """Encode a single protein sequence.

        Args:
            sequence: Protein sequence (amino acid string)

        Returns:
            Tuple of (input_ids, attention_mask)
        """
        sequence = sequence.upper()

        input_ids = [self.cls_token_id]
        for aa in sequence:
            input_ids.append(self.vocab.get(aa, self.unk_token_id))
        input_ids.append(self.eos_token_id)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]

        attention_mask = [1] * len(input_ids)

        return input_ids, attention_mask

    def encode_batch(
        self, sequences: list[str]
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Encode a batch of protein sequences with padding.

        Args:
            sequences: List of protein sequences

        Returns:
            Tuple of (input_ids, attention_masks) with padding
        """
        all_input_ids = []
        all_attention_masks = []
        max_len = 0

        for seq in sequences:
            input_ids, attention_mask = self.encode(seq)
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            max_len = max(max_len, len(input_ids))

        for i in range(len(all_input_ids)):
            padding_length = max_len - len(all_input_ids[i])
            all_input_ids[i].extend([self.pad_token_id] * padding_length)
            all_attention_masks[i].extend([0] * padding_length)

        return all_input_ids, all_attention_masks


class ProteinEmbeddingBase(ModelManagement[DenseModelDescription]):
    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        threads: int | None = None,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.threads = threads
        self._local_files_only = kwargs.pop("local_files_only", False)
        self._embedding_size: int | None = None

    def embed(
        self,
        sequences: str | Iterable[str],
        batch_size: int = 32,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        """
        Embed protein sequences.

        Args:
            sequences: Single protein sequence or iterable of sequences
            batch_size: Batch size for encoding
            parallel: Number of parallel workers (None for single-threaded)

        Yields:
            Embeddings as numpy arrays
        """
        raise NotImplementedError()

    @classmethod
    def get_embedding_size(cls, model_name: str) -> int:
        """
        Returns embedding size of the passed model.
        
        Args:
            model_name: Name of the model
        """
        descriptions = cls._list_supported_models()
        for description in descriptions:
            if description.model.lower() == model_name.lower():
                if description.dim is not None:
                    return description.dim
        raise ValueError(f"Model {model_name} not found")

    @property
    def embedding_size(self) -> int:
        """
        Returns embedding size for the current model.
        """
        if self._embedding_size is None:
            self._embedding_size = self.get_embedding_size(self.model_name)
        return self._embedding_size


class OnnxProteinModel(OnnxModel[NumpyArray]):
    """
    ONNX model handler for protein embeddings.
    """

    ONNX_OUTPUT_NAMES: list[str] | None = None

    def __init__(self) -> None:
        super().__init__()
        self.tokenizer: ProteinTokenizer | None = None

    def _load_onnx_model(
        self,
        model_dir: Path,
        model_file: str,
        threads: int | None,
        providers: Sequence[OnnxProvider] | None = None,
        cuda: bool | Device = Device.AUTO,
        device_id: int | None = None,
        extra_session_options: dict[str, Any] | None = None,
    ) -> None:
        super()._load_onnx_model(
            model_dir=model_dir,
            model_file=model_file,
            threads=threads,
            providers=providers,
            cuda=cuda,
            device_id=device_id,
            extra_session_options=extra_session_options,
        )
        vocab_path = model_dir / "vocab.txt"
        if not vocab_path.exists():
            raise ValueError(f"Could not find vocab.txt in {model_dir}")
        self.tokenizer = ProteinTokenizer(vocab_path)

    def onnx_embed(self, sequences: list[str], **kwargs: Any) -> OnnxOutputContext:
        """
        Run ONNX inference on protein sequences.

        Args:
            sequences: List of protein sequences
        Returns:
            OnnxOutputContext containing model output and inputs
        """
        assert self.tokenizer is not None

        input_ids, attention_masks = self.tokenizer.encode_batch(sequences)

        input_names = {node.name for node in self.model.get_inputs()}  # type: ignore[union-attr]
        onnx_input: dict[str, NumpyArray] = {
            "input_ids": np.array(input_ids, dtype=np.int64),
        }
        if "attention_mask" in input_names:
            onnx_input["attention_mask"] = np.array(attention_masks, dtype=np.int64)

        model_output = self.model.run(self.ONNX_OUTPUT_NAMES, onnx_input)  # type: ignore[union-attr]

        return OnnxOutputContext(
            model_output=model_output[0],
            attention_mask=np.array(attention_masks, dtype=np.int64),
            input_ids=np.array(input_ids, dtype=np.int64),
        )

    def _post_process_onnx_output(
        self, output: OnnxOutputContext, **kwargs: Any
    ) -> Iterable[NumpyArray]:
        """Convert ONNX output to embeddings with mean pooling."""
        embeddings = output.model_output
        attention_mask = output.attention_mask

        if attention_mask is None:
            raise ValueError("attention_mask is required for mean pooling")

        mask_expanded = np.expand_dims(attention_mask, axis=-1)
        sum_embeddings = np.sum(embeddings * mask_expanded, axis=1)
        sum_mask = np.sum(mask_expanded, axis=1)
        sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)
        mean_embeddings = sum_embeddings / sum_mask

        return normalize(mean_embeddings)


class OnnxProteinEmbedding(ProteinEmbeddingBase, OnnxProteinModel):
    """
    ONNX-based protein embedding implementation.
    """

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        return supported_protein_models

    def __init__(
        self,
        model_name: str = "facebook/esm2_t12_35M_UR50D",
        cache_dir: str | None = None,
        threads: int | None = None,
        providers: Sequence[OnnxProvider] | None = None,
        cuda: bool | Device = Device.AUTO,
        device_ids: list[int] | None = None,
        lazy_load: bool = False,
        device_id: int | None = None,
        specific_model_path: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(model_name, cache_dir, threads, **kwargs)
        self.providers = providers
        self.lazy_load = lazy_load
        self._extra_session_options = self._select_exposed_session_options(kwargs)
        self.device_ids = device_ids
        self.cuda = cuda

        self.device_id: int | None = None
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
        self._load_onnx_model(
            model_dir=self._model_dir,
            model_file=self.model_description.model_file,
            threads=self.threads,
            providers=self.providers,
            cuda=self.cuda,
            device_id=self.device_id,
            extra_session_options=self._extra_session_options,
        )

    def embed(
        self,
        sequences: str | Iterable[str],
        batch_size: int = 32,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        """
        Embed protein sequences.

        Args:
            sequences: Single protein sequence or iterable of sequences (amino acid strings)
            batch_size: Batch size for encoding
            parallel: Number of parallel workers (not yet supported)

        Yields:
            Embeddings as numpy arrays, one per sequence
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        if not hasattr(self, "model") or self.model is None:
            self.load_onnx_model()

        for batch in iter_batch(sequences, batch_size):
            yield from self._post_process_onnx_output(self.onnx_embed(batch, **kwargs), **kwargs)

    @classmethod
    def _get_worker_class(cls) -> Type["ProteinEmbeddingWorker"]:
        return ProteinEmbeddingWorker


class ProteinEmbeddingWorker(EmbeddingWorker[NumpyArray]):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs: Any,
    ) -> OnnxProteinEmbedding:
        return OnnxProteinEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )

    def process(
        self, items: Iterable[tuple[int, Any]]
    ) -> Iterable[tuple[int, OnnxOutputContext]]:
        for idx, batch in items:
            onnx_output = self.model.onnx_embed(batch)
            yield idx, onnx_output


class ProteinEmbedding(ProteinEmbeddingBase):
    """
    Protein sequence embedding using ESM-2 and similar models.

    Example:
        >>> from fastembed.bio import ProteinEmbedding
        >>> model = ProteinEmbedding("facebook/esm2_t12_35M_UR50D")
        >>> embeddings = list(model.embed(["MKTVRQERLKS", "GKGDPKKPRGKM"]))
        >>> print(embeddings[0].shape)
        (480,)
    """

    EMBEDDINGS_REGISTRY: list[Type[ProteinEmbeddingBase]] = [OnnxProteinEmbedding]

    @classmethod
    def list_supported_models(cls) -> list[dict[str, Any]]:
        """Lists the supported models.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing the model information.
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
        model_name: str = "facebook/esm2_t12_35M_UR50D",
        cache_dir: str | None = None,
        threads: int | None = None,
        providers: Sequence[OnnxProvider] | None = None,
        cuda: bool | Device = Device.AUTO,
        device_ids: list[int] | None = None,
        lazy_load: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize ProteinEmbedding.

        Args:
            model_name: Name of the model to use
            cache_dir: Path to cache directory
            threads: Number of threads for ONNX runtime
            providers: ONNX execution providers
            cuda: Whether to use CUDA
            device_ids: List of device IDs for multi-GPU
            lazy_load: Whether to load model lazily
        """
        super().__init__(model_name, cache_dir, threads, **kwargs)

        for EMBEDDING_MODEL_TYPE in self.EMBEDDINGS_REGISTRY:
            supported_models = EMBEDDING_MODEL_TYPE._list_supported_models()
            if any(model_name.lower() == model.model.lower() for model in supported_models):
                self.model = EMBEDDING_MODEL_TYPE(
                    model_name=model_name,
                    cache_dir=cache_dir,
                    threads=threads,
                    providers=providers,
                    cuda=cuda,
                    device_ids=device_ids,
                    lazy_load=lazy_load,
                    **kwargs,
                )
                return

        raise ValueError(
            f"Model {model_name} is not supported in ProteinEmbedding. "
            "Please check the supported models using `ProteinEmbedding.list_supported_models()`"
        )

    def embed(
        self,
        sequences: str | Iterable[str],
        batch_size: int = 32,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        """Embed protein sequences.

        Args:
            sequences: Single protein sequence or iterable of sequences (amino acid strings)
            batch_size: Batch size for encoding
            parallel: Number of parallel workers

        Yields:
            Embeddings as numpy arrays, one per sequence
        """
        yield from self.model.embed(sequences, batch_size, parallel, **kwargs)