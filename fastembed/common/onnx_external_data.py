"""Loading ONNX models with external data from a huggingface_hub cache.

onnxruntime>=1.24 refuses external data that, once symlinks are resolved, is located outside of
the directory of the model file, and of the directory the model file resolves to (a fallback of
1.24.2, pyproject.toml excludes 1.24.0 and 1.24.1). Since huggingface_hub 1.32, a snapshot is made
of symlinks into a blob store shared by the whole cache and sharded by hash, where a model and its
data may resolve into different directories. Then the model and its data are hardlinked into
`ONNX_SNAPSHOTS_DIR` of the repo cache, laid out as in the snapshot, which takes no extra space.
"""

import contextlib
import os
import shutil
from pathlib import Path

# Next to `snapshots` in the cache of a huggingface_hub repo: `onnx_snapshots/<revision>/...`
ONNX_SNAPSHOTS_DIR = "onnx_snapshots"


def link_external_data(model_dir: Path, model_file: str, additional_files: list[str]) -> Path:
    """Returns a path of `model_file` from which onnxruntime can load its external data.

    Args:
        model_dir (Path): The directory with the model files, e.g. a huggingface_hub snapshot.
        model_file (str): The path of the ONNX file, relative to `model_dir`.
        additional_files (list[str]): Other files of the model, relative to `model_dir`,
            among which its external data.

    Returns:
        Path: The ONNX file to load, `model_dir / model_file` unless it had to be linked.

    Raises:
        OSError: If the files had to be linked, but couldn't be, e.g. in a read-only cache or
            on a filesystem without hardlinks.
    """
    model_path = model_dir / model_file
    snapshots_dir = model_dir.parent
    repo_dir = snapshots_dir.parent
    if snapshots_dir.name != "snapshots" or not repo_dir.name.startswith("models--"):
        return model_path  # not a huggingface_hub cache, there is no place of ours to link into
    if not model_path.exists():
        return model_path  # onnxruntime reports it more clearly than a failed hardlink would

    # external data is located relative to the model file, so it can't be anywhere else
    data_paths = [
        path
        for path in (model_dir / file for file in additional_files)
        if path.parent.is_relative_to(model_path.parent) and path.exists()
    ]
    # onnxruntime accepts data within these, as in the caches of older huggingface_hub versions
    real_model_dirs = (
        os.path.realpath(model_path.parent),
        os.path.dirname(os.path.realpath(model_path)),
    )
    if all(
        any(Path(os.path.realpath(path)).is_relative_to(d) for d in real_model_dirs)
        for path in data_paths
    ):
        return model_path

    links_dir = repo_dir / ONNX_SNAPSHOTS_DIR
    for path in (model_path, *data_paths):
        _link_file(path, links_dir / model_dir.name / path.relative_to(model_dir))

    # the links keep the blobs on disk, so drop those of revisions deleted from the cache
    for revision_dir in links_dir.iterdir():
        if not (snapshots_dir / revision_dir.name).is_dir():
            shutil.rmtree(revision_dir, ignore_errors=True)

    return links_dir / model_dir.name / model_file


def _link_file(source: Path, link: Path) -> None:
    """Makes `link` a hardlink to the file `source` resolves to, unless it already is one."""
    target = os.path.realpath(source)
    try:
        if os.path.samefile(link, target):
            return
        # a link to a blob downloaded again since, or a copy of the cache without its hardlinks
        link.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        # e.g. such a copy on a read-only filesystem, which works all the same
        if os.path.getsize(link) != os.path.getsize(target):
            raise
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    # os.link is atomic, so if the link exists, another process loading the model just made it
    with contextlib.suppress(FileExistsError):
        os.link(target, link)
