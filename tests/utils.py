import shutil

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
    if isinstance(model_dir, str):
        model_dir = Path(model_dir)

    if model_dir.parent.parent.name.startswith("models--"):
        model_dir = model_dir.parent.parent

    if model_dir.exists():
        try:
            shutil.rmtree(model_dir)
        except Exception as e:
            print(e)
            print("sleeping for 3 seconds...")
            import time

            time.sleep(3)
            print("trying out again")

            if model_dir.exists():
                # shutil.rmtree(model_dir)
                shutil.rmtree(model_dir / "refs")
                shutil.rmtree(model_dir / "snapshots")
                shutil.rmtree(model_dir / "blobs")
