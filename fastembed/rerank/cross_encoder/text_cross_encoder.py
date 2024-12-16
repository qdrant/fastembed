from typing import Any, Iterable, Optional, Sequence, Type

from fastembed.common import OnnxProvider
from fastembed.rerank.cross_encoder.onnx_text_cross_encoder import OnnxTextCrossEncoder
from fastembed.rerank.cross_encoder.text_cross_encoder_base import TextCrossEncoderBase


class TextCrossEncoder(TextCrossEncoderBase):
    CROSS_ENCODER_REGISTRY: list[Type[TextCrossEncoderBase]] = [
        OnnxTextCrossEncoder,
    ]

    @classmethod
    def list_supported_models(cls) -> list[dict[str, Any]]:
        """Lists the supported models.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing the model information.

            Example:
                ```
                [
                    {
                        "model": "Xenova/ms-marco-MiniLM-L-6-v2",
                        "size_in_GB": 0.08,
                        "sources": {
                            "hf": "Xenova/ms-marco-MiniLM-L-6-v2",
                        },
                        "model_file": "onnx/model.onnx",
                        "description": "MiniLM-L-6-v2 model optimized for re-ranking tasks.",
                        "license": "apache-2.0",
                    }
                ]
                ```
        """
        result = []
        for encoder in cls.CROSS_ENCODER_REGISTRY:
            result.extend(encoder.list_supported_models())
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

        for CROSS_ENCODER_TYPE in self.CROSS_ENCODER_REGISTRY:
            supported_models = CROSS_ENCODER_TYPE.list_supported_models()
            if any(model_name.lower() == model["model"].lower() for model in supported_models):
                self.model = CROSS_ENCODER_TYPE(
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
            f"Model {model_name} is not supported in TextCrossEncoder."
            "Please check the supported models using `TextCrossEncoder.list_supported_models()`"
        )

    def rerank(
        self, query: str, documents: Iterable[str], batch_size: int = 64, **kwargs: Any
    ) -> Iterable[float]:
        """Rerank a list of documents based on a query.

        Args:
            query: Query to rerank the documents against
            documents: Iterator of documents to rerank
            batch_size: Batch size for reranking

        Returns:
            Iterable of scores for each document
        """
        yield from self.model.rerank(query, documents, batch_size=batch_size, **kwargs)

    def rerank_pairs(
        self,
        pairs: Iterable[tuple[str, str]],
        batch_size: int = 64,
        parallel: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterable[float]:
        """
        Rerank a list of query-document pairs.

        Args:
            pairs (Iterable[tuple[str, str]]): An iterable of tuples, where each tuple contains a query and a document
                to be scored together.
            batch_size (int, optional): The number of query-document pairs to process in a single batch. Defaults to 64.
            parallel (Optional[int], optional): The number of parallel processes to use for reranking.
                If None, parallelization is disabled. Defaults to None.
            **kwargs (Any): Additional arguments to pass to the underlying reranking model.

        Returns:
            Iterable[float]: An iterable of scores corresponding to each query-document pair in the input.
            Higher scores indicate a stronger match between the query and the document.

        Example:
            >>> encoder = TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")
            >>> pairs = [("What is AI?", "Artificial intelligence is ..."), ("What is ML?", "Machine learning is ...")]
            >>> scores = list(encoder.rerank_pairs(pairs))
            >>> print(list(map(lambda x: round(x, 2), scores)))
            [-1.24, -10.6]
        """
        yield from self.model.rerank_pairs(
            pairs, batch_size=batch_size, parallel=parallel, **kwargs
        )
