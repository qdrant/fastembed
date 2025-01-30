import shutil
import traceback

from pathlib import Path
from types import TracebackType
from typing import Union, Callable, Any, Type


def delete_model_cache(model_dir: Union[str, Path]) -> None:
    """Delete the model cache directory.

    If a model was downloaded from the HuggingFace model hub, then _model_dir is the dir to snapshots, removing
    it won't help to release the memory, because data is in blobs directory.
    If a model was downloaded from GCS, then we can just remove model_dir

    Args:
        model_dir (Union[str, Path]): The path to the model cache directory.
    """

    def on_error(
        func: Callable[..., Any],
        path: str,
        exc_info: tuple[Type[BaseException], BaseException, TracebackType],
    ) -> None:
        print("Failed to remove: ", path)
        print("Exception: ", exc_info)
        traceback.print_exception(*exc_info)

    if isinstance(model_dir, str):
        model_dir = Path(model_dir)

    if model_dir.parent.parent.name.startswith("models--"):
        model_dir = model_dir.parent.parent

    if model_dir.exists():
        # todo: PermissionDenied is raised on blobs removal in Windows, with blobs > 2GB
        shutil.rmtree(model_dir, onerror=on_error)
