from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fastembed.image.transform.functional import normalize, resize
from fastembed.image.transform.operators import (
    CenterCrop,
    Compose,
    ConvertToRGB,
    Resize,
    ResizeLongestEdge,
    Transform,
)


def test_center_crop_odd_padding_keeps_batch_shape_and_pixels() -> None:
    pixels = np.arange(1, 28, dtype=np.uint8).reshape(3, 3, 3)
    processor = Compose.from_config(
        {
            "do_resize": False,
            "do_center_crop": True,
            "crop_size": 4,
            "do_rescale": False,
        }
    )
    images = [Image.fromarray(pixels), Image.new("RGB", (4, 4))]

    batch = np.array(processor(images))
    expected = np.zeros((3, 4, 4), dtype=np.float32)
    expected[:, 1:, 1:] = pixels.transpose(2, 0, 1)

    assert batch.shape == (2, 3, 4, 4)
    np.testing.assert_array_equal(batch[0], expected)


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


@pytest.mark.parametrize(
    ("mean", "std"),
    [
        ([0.1, 0.2, 0.3], [0.5, 0.6, 0.7]),  # per-channel, as every model config gives it
        (0.5, 0.25),  # scalar, expanded to one value per channel
    ],
)
def test_normalize_chw_is_channel_wise(
    mean: list[float] | float, std: list[float] | float
) -> None:
    """Each channel must be normalized by its own mean/std, not by any other axis."""
    rng = np.random.default_rng(0)
    image = rng.random((3, 5, 7)).astype(np.float32)
    means = mean if isinstance(mean, list) else [mean] * 3
    stds = std if isinstance(std, list) else [std] * 3

    result = normalize(image, mean=mean, std=std)

    for c in range(3):
        assert np.allclose(result[c], (image[c] - means[c]) / stds[c], atol=1e-6)


@pytest.mark.parametrize("batch_size", [2, 3])
def test_normalize_batched_matches_per_image(batch_size: int) -> None:
    """A batch must give exactly what the (C, H, W) path gives image by image.

    batch_size 2 used to raise, since transposing reversed every axis; batch_size 3
    matched the channel count and silently normalized along the batch axis instead.
    """
    rng = np.random.default_rng(2)
    batch = rng.random((batch_size, 3, 4, 4)).astype(np.float32)
    mean, std = [0.1, 0.2, 0.3], [0.5, 0.6, 0.7]

    result = normalize(batch, mean=mean, std=std)

    per_image = np.stack([normalize(image, mean=mean, std=std) for image in batch])
    assert result.shape == batch.shape
    assert np.array_equal(result, per_image)


def test_normalize_rejects_input_without_a_channel_axis() -> None:
    """Every pipeline runs ConvertToRGB first, so normalize only ever sees (C, H, W)."""
    with pytest.raises(ValueError, match=r"must be \(C, H, W\)"):
        normalize(np.zeros((4, 6), dtype=np.float32), mean=0.5, std=0.25)


@pytest.mark.parametrize("shape", [(3, 2, 2), (2, 3, 2, 2)])
@pytest.mark.parametrize("std", [0.0, [0.5, -0.0, 0.5], [0.5, 1e-50, 0.5]])
def test_normalize_rejects_zero_standard_deviation(
    shape: tuple[int, ...], std: float | list[float]
) -> None:
    image = np.ones(shape, dtype=np.float32)
    with np.errstate(divide="raise", invalid="raise"):
        with pytest.raises(ValueError, match="std.*zero"):
            normalize(image, mean=0.5, std=std)


def test_processor_rejects_zero_standard_deviation() -> None:
    processor = Compose.from_config(
        {
            "do_resize": False,
            "do_center_crop": False,
            "do_normalize": True,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.0, 0.5],
        }
    )
    with np.errstate(divide="raise", invalid="raise"):
        with pytest.raises(ValueError, match="std.*zero"):
            processor([Image.new("RGB", (2, 2), color="white")])


def test_normalize_preserves_small_nonzero_standard_deviation() -> None:
    image = np.ones((3, 2, 2), dtype=np.float32)
    result = normalize(image, mean=0.0, std=1e-10)
    np.testing.assert_allclose(result, np.full_like(image, 1e10))
    np.testing.assert_array_equal(image, np.ones_like(image))


@pytest.mark.parametrize(
    ("mean", "std", "expected"),
    [
        ([0.1, 0.2], [1.0, 1.0, 1.0], "mean must"),
        ([0.1, 0.2, 0.3], [1.0, 1.0], "std must"),
    ],
)
def test_normalize_channel_count_mismatch_raises(
    mean: list[float], std: list[float], expected: str
) -> None:
    image = np.zeros((3, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match=expected):
        normalize(image, mean=mean, std=std)


@pytest.mark.parametrize(
    ("first", "expected"),
    [
        (Resize(size=224), (800, 600)),  # 1/2 scale keeps 2x the shortest edge
        (Resize(size=(224, 224)), (800, 600)),
        (ResizeLongestEdge(size=256), (800, 600)),
        (Resize(size=448), (1600, 1200)),  # 1/2 scale would go below 2x
        (CenterCrop(size=(224, 224)), (1600, 1200)),  # no resize to go by
    ],
)
def test_open_image_decodes_large_jpegs_at_a_reduced_scale(
    tmp_path: Path, first: Transform, expected: tuple[int, int]
) -> None:
    path = tmp_path / "photo.jpeg"
    Image.new("RGB", (1600, 1200)).save(path)
    with Compose([ConvertToRGB(), first]).open_image(path) as image:
        assert image.size == expected
