import numpy as np
import pytest
from PIL import Image

from fastembed.image.transform.functional import normalize, resize
from fastembed.image.transform.operators import Compose


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


@pytest.mark.parametrize("splitting", [False, True])
@pytest.mark.parametrize("rescale_and_normalize", [False, True])
def test_compose_grid_metadata_keeps_default_outputs(splitting, rescale_and_normalize) -> None:
    processor = Compose.from_config(
        {
            "image_processor_type": "Idefics3ImageProcessor",
            "do_resize": False,
            "do_image_splitting": splitting,
            "max_image_size": {"longest_edge": 4},
            "do_rescale": rescale_and_normalize,
            "do_normalize": rescale_and_normalize,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        }
    )
    images = [Image.new("RGB", size, (32, 64, 128)) for size in [(4, 8), (8, 4), (4, 4)]]
    ordinary = processor(images)
    metadata = {"existing": "preserved"}
    with_metadata = processor(images, metadata=metadata)

    assert isinstance(ordinary, list) and isinstance(with_metadata, list)
    assert metadata["image_grid"] == ([(2, 1), (1, 2), (0, 0)] if splitting else [(0, 0)] * 3)
    assert metadata["existing"] == "preserved"
    for expected, actual in zip(ordinary, with_metadata):
        assert isinstance(expected, list) and isinstance(actual, list)
        assert len(expected) == len(actual)
        for expected_patch, actual_patch in zip(expected, actual):
            assert isinstance(expected_patch, np.ndarray) and isinstance(actual_patch, np.ndarray)
            np.testing.assert_array_equal(actual_patch, expected_patch)

    # Geometry belongs to each call, including repeated use of a metadata dictionary.
    processor([images[-1]], metadata=metadata)
    assert metadata["image_grid"] == [(0, 0)]


def test_splitter_grid_keeps_row_major_pixels_and_global_last() -> None:
    from fastembed.image.transform.operators import ImageSplitter

    image = np.arange(3 * 8 * 12, dtype=np.float32).reshape(3, 8, 12)
    splitter = ImageSplitter(max_size=4)
    ordinary = splitter([image])
    patches, grids = splitter.split_with_grid([image])

    assert grids == [(2, 3)]
    assert isinstance(ordinary, list) and isinstance(ordinary[0], list)
    assert len(patches[0]) == len(ordinary[0]) == 7
    for row in range(2):
        for col in range(3):
            index = row * 3 + col
            np.testing.assert_array_equal(
                patches[0][index], image[:, row * 4 : (row + 1) * 4, col * 4 : (col + 1) * 4]
            )
    for actual, expected in zip(patches[0], ordinary[0]):
        np.testing.assert_array_equal(actual, expected)
    assert patches[0][-1].shape == (3, 4, 4)


def test_compose_metadata_does_not_change_flat_image_batches() -> None:
    processor = Compose.from_config({"do_resize": False, "do_rescale": False})
    images = [Image.new("RGB", (4, 4))]
    metadata = {}
    actual = processor(images, metadata=metadata)

    assert isinstance(actual, list) and isinstance(actual[0], np.ndarray)
    assert metadata == {"image_grid": [(0, 0)]}
    np.testing.assert_array_equal(actual, processor(images))
