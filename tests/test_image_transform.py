import numpy as np
import pytest
from PIL import Image

from fastembed.image.transform.functional import normalize, resize


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ((100, 200), (200, 100)),  # the bug: a non-square size came back transposed
        ((224, 224), (224, 224)),  # the square path every shipped model takes
    ],
)
def test_resize_tuple_is_height_width(size: tuple[int, int], expected: tuple[int, int]) -> None:
    """A ``(height, width)`` size must reach Pillow as ``(width, height)``."""
    resized = resize(Image.new("RGB", (300, 300)), size=size)

    assert resized.size == expected  # PIL reports (width, height)


def test_resize_int_keeps_shortest_edge_behaviour() -> None:
    """The int branch already emitted Pillow order; it must not be disturbed."""
    landscape = Image.new("RGB", (400, 200))
    portrait = Image.new("RGB", (200, 400))

    # size sets the shortest edge, and the aspect ratio is preserved.
    assert resize(landscape, size=100).size == (200, 100)
    assert resize(portrait, size=100).size == (100, 200)


def _reference_normalize(image, mean, std):
    """Channel-wise normalization with explicit broadcasting, used as ground truth."""
    channel_axis = 1 if image.ndim == 4 else 0
    shape = [1] * image.ndim
    shape[channel_axis] = image.shape[channel_axis]
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(shape)
    std_arr = np.asarray(std, dtype=np.float32).reshape(shape)
    return (image.astype(np.float32) - mean_arr) / std_arr


def test_normalize_chw_matches_channel_wise():
    rng = np.random.default_rng(0)
    image = rng.random((3, 5, 7)).astype(np.float32)
    mean, std = [0.1, 0.2, 0.3], [0.5, 0.6, 0.7]

    result = normalize(image, mean=mean, std=std)

    assert np.allclose(result, _reference_normalize(image, mean, std), atol=1e-6)


def test_normalize_scalar_mean_std():
    rng = np.random.default_rng(1)
    image = rng.random((3, 4, 4)).astype(np.float32)

    result = normalize(image, mean=0.5, std=0.25)

    assert np.allclose(result, (image - 0.5) / 0.25, atol=1e-6)


def test_normalize_batched_input_normalizes_per_channel():
    # (N, C, H, W): every channel c is filled with the constant c, so subtracting
    # mean == c and dividing by 1 must yield all zeros regardless of batch size.
    image = np.zeros((3, 3, 2, 2), dtype=np.float32)
    for c in range(3):
        image[:, c] = c

    result = normalize(image, mean=[0.0, 1.0, 2.0], std=[1.0, 1.0, 1.0])

    assert np.allclose(result, 0.0)


def test_normalize_batched_input_when_batch_differs_from_channels():
    # N != C used to raise because transposing reversed every axis.
    rng = np.random.default_rng(2)
    image = rng.random((2, 3, 4, 4)).astype(np.float32)
    mean, std = [0.1, 0.2, 0.3], [0.5, 0.6, 0.7]

    result = normalize(image, mean=mean, std=std)

    assert result.shape == image.shape
    assert np.allclose(result, _reference_normalize(image, mean, std), atol=1e-6)


def test_normalize_channel_count_mismatch_raises():
    image = np.zeros((3, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        normalize(image, mean=[0.1, 0.2], std=[1.0, 1.0, 1.0])
