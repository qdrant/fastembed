import shutil
import traceback

from pathlib import Path
from typing import Union


def delete_model_cache(model_dir: Union[str, Path]) -> None:
    """Delete the model cache directory.

    If a model was downloaded from the HuggingFace model hub, then _model_dir is the dir to snapshots, removing
    it won't help to release the memory, because data is in blobs directory.
    If a model was downloaded from GCS, then we can just remove model_dir

    Args:
        model_dir (Union[str, Path]): The path to the model cache directory.
    """
    import os

    def recursive_chmod(path: str, mode: int):
        """
        Recursively change permissions of files and directories.

        :param path: Path to the file or directory.
        :param mode: File mode (e.g., 0o755).
        """
        # Change the permission of the current path
        print("change permission", path, mode)
        if os.path.exists(str(path)):
            print("path exist", path)
            p = Path(path)
            if p.parent.exists() and p.parent.is_dir():
                print("parent content: ", list(p.parent.iterdir()))
            os.chmod(str(path), mode)

            # If the path is a directory, recursively apply chmod to its contents
            if os.path.isdir(str(path)):
                for entry in os.listdir(str(path)):
                    full_path = os.path.join(str(path), entry)
                    recursive_chmod(full_path, mode)

    def on_error(func, path, exc_info):
        if not Path(path).exists():
            print("path does not exist", path)
            return
        exc_type, exc_value, exc_traceback = exc_info

        # Example usage:
        # Change permissions to 755 recursively for a directory
        def last_resort(func, path, exc_info):
            print("Failed to remove: ", path)
            print("Exception: ", exc_value)
            print("Traceback: ", traceback.format_tb(exc_traceback))

        recursive_chmod(path, 0o755)
        shutil.rmtree(path, onerror=last_resort)

    if isinstance(model_dir, str):
        model_dir = Path(model_dir)

    if model_dir.parent.parent.name.startswith("models--"):
        model_dir = model_dir.parent.parent

    if model_dir.exists():
        shutil.rmtree(model_dir, onerror=on_error)
