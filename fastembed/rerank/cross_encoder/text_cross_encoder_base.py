from typing import Iterable, Optional, Union

from fastembed.common.model_management import ModelManagement


class TextCrossEncoderBase(ModelManagement):
    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.threads = threads
        self._local_files_only = kwargs.pop("local_files_only", False)

    def rerank(
        self,
        query: str,
        documents: Union[str, Iterable[str]],
        batch_size: int = 64,
        **kwargs,
    ) -> Iterable[float]:
        raise NotImplementedError("This method should be overridden by subclasses")
