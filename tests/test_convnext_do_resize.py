import numpy as np
import pytest
from PIL import Image

from fastembed.image.transform.operators import Compose


def _patterned_rgb(width: int, height: int) -> Image.Image:
    pixels = np.random.default_rng(0).integers(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


@pytest.mark.parametrize("shortest_edge", [224, 384, 512])
@pytest.mark.parametrize("image_size", [(67, 43), (43, 67), (224, 224)])
def test_convnext_disabled_resize_preserves_shape_and_pixels(
    shortest_edge: int, image_size: tuple[int, int]
) -> None:
    image = _patterned_rgb(*image_size)
    processor = Compose.from_config(
        {
            "image_processor_type": "ConvNextFeatureExtractor",
            "do_resize": False,
            "size": {"shortest_edge": shortest_edge},
            "do_rescale": False,
        }
    )

    output = processor([image])[0]

    expected = np.asarray(image).transpose(2, 0, 1)
    assert output.shape == expected.shape
    np.testing.assert_array_equal(output, expected)


def test_convnext_disabled_resize_does_not_require_geometry_settings() -> None:
    image = _patterned_rgb(13, 7)
    processor = Compose.from_config(
        {
            "image_processor_type": "ConvNextFeatureExtractor",
            "do_resize": False,
            "do_rescale": False,
        }
    )

    np.testing.assert_array_equal(processor([image])[0], np.asarray(image).transpose(2, 0, 1))


@pytest.mark.parametrize("do_resize", [True, None], ids=["enabled", "omitted"])
@pytest.mark.parametrize(
    ("shortest_edge", "resize_size", "crop_box"),
    [
        (224, (384, 256), (80, 16, 304, 240)),
        (384, (384, 384), None),
        (512, (512, 512), None),
    ],
)
def test_convnext_enabled_and_default_resize_preserve_existing_behavior(
    do_resize: bool | None,
    shortest_edge: int,
    resize_size: tuple[int, int],
    crop_box: tuple[int, int, int, int] | None,
) -> None:
    image = _patterned_rgb(300, 200)
    config = {
        "image_processor_type": "ConvNextFeatureExtractor",
        "size": {"shortest_edge": shortest_edge},
        "do_rescale": False,
    }
    if do_resize is not None:
        config["do_resize"] = do_resize
    processor = Compose.from_config(config)
    expected = image.resize(resize_size, Image.Resampling.BICUBIC)
    if crop_box is not None:
        expected = expected.crop(crop_box)

    np.testing.assert_array_equal(processor([image])[0], np.asarray(expected).transpose(2, 0, 1))


@pytest.mark.parametrize("do_resize", [True, None], ids=["enabled", "omitted"])
def test_convnext_enabled_resize_still_rejects_invalid_size(do_resize: bool | None) -> None:
    config = {
        "image_processor_type": "ConvNextFeatureExtractor",
        "size": {"height": 224, "width": 224},
    }
    if do_resize is not None:
        config["do_resize"] = do_resize

    with pytest.raises(ValueError, match="shortest_edge"):
        Compose.from_config(config)


def test_convnext_disabled_resize_still_rescales_and_normalizes() -> None:
    image = _patterned_rgb(13, 7)
    processor = Compose.from_config(
        {
            "image_processor_type": "ConvNextFeatureExtractor",
            "do_resize": False,
            "size": {"shortest_edge": 224},
            "do_rescale": True,
            "rescale_factor": 0.25,
            "do_normalize": True,
            "image_mean": [1.0, 2.0, 3.0],
            "image_std": [2.0, 4.0, 8.0],
        }
    )
    pixels = np.asarray(image).transpose(2, 0, 1).astype(np.float32)
    expected = (pixels * 0.25 - np.array([1.0, 2.0, 3.0])[:, None, None]) / np.array(
        [2.0, 4.0, 8.0]
    )[:, None, None]

    np.testing.assert_array_equal(processor([image])[0], expected)


@pytest.mark.parametrize("mode", ["CLIPImageProcessor", "SiglipImageProcessor"])
@pytest.mark.parametrize("do_resize", [False, True, None], ids=["disabled", "enabled", "omitted"])
def test_other_processors_keep_existing_resize_flag_behavior(
    mode: str, do_resize: bool | None
) -> None:
    image = _patterned_rgb(67, 43)
    config = {
        "image_processor_type": mode,
        "size": {"height": 32, "width": 48},
        "do_rescale": False,
    }
    if do_resize is not None:
        config["do_resize"] = do_resize
    expected = image.resize((48, 32), Image.Resampling.BICUBIC) if do_resize else image

    processor = Compose.from_config(config)

    np.testing.assert_array_equal(processor([image])[0], np.asarray(expected).transpose(2, 0, 1))
