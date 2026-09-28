import numpy as np
import pytest
from PIL import Image

from fastembed.image.transform.functional import center_crop, normalize, resize
from fastembed.image.transform.operators import Compose


@pytest.mark.parametrize("as_array", [False, True], ids=["pil", "numpy"])
@pytest.mark.parametrize(
    ("image_size", "crop_size"),
    [
        ((3, 3), (4, 4)),  # odd padding on both axes
        ((3, 7), (4, 4)),  # pad height, crop width
        ((7, 3), (4, 4)),  # crop height, pad width
        ((1, 3), (6, 8)),  # more than one padding pixel on each side
        ((3, 6), (4, 6)),  # pad height only
        ((4, 5), (4, 6)),  # pad width only
        ((2, 4), (4, 6)),  # even padding
        ((6, 8), (4, 6)),  # no padding needed
        ((4, 6), (4, 6)),  # already the requested size
    ],
)
def test_center_crop_shape_and_pixels(
    image_size: tuple[int, int], crop_size: tuple[int, int], as_array: bool
) -> None:
    height, width = image_size
    pixels = np.arange(1, height * width * 3 + 1, dtype=np.uint8).reshape(height, width, 3)
    pil_image = Image.fromarray(pixels)
    image = pixels.transpose(2, 0, 1) if as_array else pil_image

    crop_height, crop_width = crop_size
    top, left = (height - crop_height) // 2, (width - crop_width) // 2
    # Pillow independently supplies zero padding outside the requested crop box.
    expected = np.asarray(
        pil_image.crop((left, top, left + crop_width, top + crop_height))
    ).transpose(2, 0, 1)

    result = center_crop(image, size=crop_size)

    assert result.shape == (3, crop_height, crop_width)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("rectangular", [False, True])
def test_center_crop_batches_different_image_sizes(rectangular: bool) -> None:
    height, width = 4, 6 if rectangular else 4
    processor = Compose.from_config(
        {
            "do_resize": False,
            "do_center_crop": True,
            "crop_size": {"height": height, "width": width} if rectangular else height,
            "do_rescale": True,
            "rescale_factor": 1 / 255,
        }
    )
    images = [
        Image.new("RGB", (3, 3), color=(10, 20, 30)),
        Image.new("RGB", (width, height), color=(10, 20, 30)),
        Image.new("RGB", (9, 7), color=(10, 20, 30)),
    ]

    # OnnxImageModel also constructs its input batch with np.array(processor(images)).
    batch = np.array(processor(images))

    assert batch.shape == (len(images), 3, height, width)
    assert batch.dtype == np.float32
    np.testing.assert_array_equal(batch[0, :, 0, :], 0)
    np.testing.assert_allclose(batch[0, :, height // 2, width // 2], np.array([10, 20, 30]) / 255)


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
