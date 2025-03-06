import shutil
import traceback

from pathlib import Path
from types import TracebackType
from typing import Union, Callable, Any, Type, Optional

from fastembed.common.model_description import BaseModelDescription


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


def should_test_model(
    model_desc: BaseModelDescription,
    autotest_model_name: str,
    is_ci: Optional[str],
    is_manual: bool,
):
    """Determine if a model should be tested based on environment

    Tests can be run either in ci or locally.
    Testing all models each time in ci is too long.
    The testing scheme in ci and on a local machine are different, therefore, there are 3 possible scenarious.
    1) Run lightweight tests in ci:
        - test only one model that has been manually chosen as a representative for a certain class family
    2) Run heavyweight (manual) tests in ci:
        - test all models
        Running tests in ci each time is too expensive, however, it's fine to run it one time with a manual dispatch
    3) Run tests locally:
        - test all models, which are not too heavy, since network speed might be a bottleneck

    """
    if not is_ci:
        if model_desc.size_in_GB > 1:
            return False
    elif not is_manual and model_desc.model != autotest_model_name:
        return False
    return True
