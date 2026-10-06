import numpy as np
import pytest
from PIL import Image

from fastembed.image.transform.functional import crop_ndarray, pad2square, resize_ndarray
from fastembed.image.transform.operators import (
    ImageSplitter,
    ResizeForVisionEncoder,
    SquareResize,
)


def _column_index_image(width: int, height: int) -> Image.Image:
    """RGB image whose every pixel holds its own column index, so crops are easy to locate."""
    row = np.arange(width, dtype=np.uint8)
    pixels = np.broadcast_to(row[None, :, None], (height, width, 3))
    return Image.fromarray(np.ascontiguousarray(pixels))


def test_pad2square_pads_smaller_image_with_fill_color() -> None:
    image = Image.new("RGB", (3, 2), color=(255, 0, 0))

    padded = np.asarray(pad2square(image, size=4, fill_color=(0, 0, 255)))

    assert padded.shape == (4, 4, 3)
    # the image is pasted at the top-left corner, the rest keeps the fill color
    np.testing.assert_array_equal(padded[:2, :3], np.full((2, 3, 3), (255, 0, 0)))
    np.testing.assert_array_equal(padded[2:, :], np.full((2, 4, 3), (0, 0, 255)))
    np.testing.assert_array_equal(padded[:, 3:], np.full((4, 1, 3), (0, 0, 255)))


@pytest.mark.parametrize(
    ("width", "expected_left"),
    [
        (7, 2),  # (7 - 4) / 2 = 1.5 rounds half to even: 2
        (9, 2),  # (9 - 4) / 2 = 2.5 rounds half to even: 2
        (8, 2),
    ],
)
def test_pad2square_center_crops_larger_image_like_torchvision(
    width: int, expected_left: int
) -> None:
    image = _column_index_image(width=width, height=4)

    cropped = np.asarray(pad2square(image, size=4))

    assert cropped.shape == (4, 4, 3)
    np.testing.assert_array_equal(cropped[0, :, 0], np.arange(expected_left, expected_left + 4))


def test_crop_ndarray_channel_first_and_last_agree() -> None:
    rng = np.random.default_rng(0)
    chw = rng.integers(0, 256, size=(3, 5, 7), dtype=np.uint8)
    hwc = chw.transpose(1, 2, 0)

    cropped_chw = crop_ndarray(chw, x1=1, y1=2, x2=5, y2=4, channel_first=True)
    cropped_hwc = crop_ndarray(hwc, x1=1, y1=2, x2=5, y2=4, channel_first=False)

    assert cropped_chw.shape == (3, 2, 4)
    np.testing.assert_array_equal(cropped_chw, chw[:, 2:4, 1:5])
    np.testing.assert_array_equal(cropped_hwc.transpose(2, 0, 1), cropped_chw)


def test_resize_ndarray_size_is_width_height() -> None:
    image = np.zeros((3, 10, 10), dtype=np.uint8)

    resized = resize_ndarray(image, size=(6, 4))

    # PIL takes (width, height); the result keeps the (C, H, W) layout
    assert resized.shape == (3, 4, 6)
    assert resized.dtype == np.uint8


def test_resize_ndarray_keeps_float_images_in_unit_range() -> None:
    image = np.full((3, 8, 8), 0.5, dtype=np.float32)

    resized = resize_ndarray(image, size=(4, 4))

    assert resized.dtype == np.float32
    assert resized.shape == (3, 4, 4)
    # scaled to uint8 for PIL and back, so values are quantised to multiples of 1/255
    np.testing.assert_allclose(resized, np.full((3, 4, 4), 127 / 255), atol=1e-6)


def test_resize_ndarray_channel_last() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    resized = resize_ndarray(image, size=(6, 4), channel_first=False)

    assert resized.shape == (4, 6, 3)


@pytest.mark.parametrize(
    ("height", "width", "expected_hw"),
    [
        (
            600,
            1000,
            (1024, 1024),
        ),  # landscape: width 1024, height int(1024 / (1000 / 600)) = 614 -> 1024
        (1000, 300, (1024, 512)),  # portrait: height 1024, width int(1024 * 0.3) = 307 -> 512
        (50, 100, (512, 512)),  # smaller than one tile: rounded up to a single tile
        (512, 512, (512, 512)),  # already a multiple of the tile size
    ],
)
def test_resize_for_vision_encoder_rounds_up_to_tile_multiples(
    height: int, width: int, expected_hw: tuple[int, int]
) -> None:
    image = np.zeros((3, height, width), dtype=np.uint8)

    (resized,) = ResizeForVisionEncoder(max_size=512)([image])

    assert resized.shape == (3, *expected_hw)


def test_image_splitter_returns_small_image_unchanged() -> None:
    image = np.zeros((3, 100, 200), dtype=np.uint8)

    (frames,) = ImageSplitter(max_size=512)([image])

    assert len(frames) == 1
    assert frames[0] is image


@pytest.mark.parametrize(
    ("height", "width", "grid"),
    [
        # ResizeForVisionEncoder runs first, so the splitter only sees multiples of the tile size
        (1024, 512, (2, 1)),
        (512, 1024, (1, 2)),
        (1024, 1536, (2, 3)),
    ],
)
def test_image_splitter_patches_cover_image_in_row_order(
    height: int, width: int, grid: tuple[int, int]
) -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(3, height, width), dtype=np.uint8)
    rows, cols = grid

    (frames,) = ImageSplitter(max_size=512)([image])

    # rows * cols local 512 x 512 tiles in row order, then the global view
    assert len(frames) == rows * cols + 1
    for r in range(rows):
        for c in range(cols):
            patch = frames[r * cols + c]
            expected = image[:, r * 512 : (r + 1) * 512, c * 512 : (c + 1) * 512]
            np.testing.assert_array_equal(patch, expected)
    assert frames[-1].shape == (3, 512, 512)


def test_image_splitter_keeps_images_grouped() -> None:
    small = np.zeros((3, 64, 64), dtype=np.uint8)
    large = np.zeros((3, 1024, 1024), dtype=np.uint8)

    result = ImageSplitter(max_size=512)([small, large, small])

    assert [len(frames) for frames in result] == [1, 5, 1]


def test_square_resize_wraps_each_image_in_a_list() -> None:
    images = [np.zeros((3, 100, 300), dtype=np.uint8), np.zeros((3, 40, 20), dtype=np.uint8)]

    result = SquareResize(size=64)(images)

    assert len(result) == 2
    for frames in result:
        assert len(frames) == 1
        assert frames[0].shape == (3, 64, 64)
