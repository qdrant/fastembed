from typing import Union

import numpy as np
from PIL import Image

from fastembed.common.types import NumpyArray


def convert_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image

    image = image.convert("RGB")
    return image


def center_crop(
    image: Union[Image.Image, NumpyArray],
    size: tuple[int, int],
) -> NumpyArray:
    if isinstance(image, np.ndarray):
        _, orig_height, orig_width = image.shape
    else:
        orig_height, orig_width = image.height, image.width
        # (H, W, C) -> (C, H, W)
        image = np.array(image).transpose((2, 0, 1))

    crop_height, crop_width = size

    # left upper corner (0, 0)
    top = (orig_height - crop_height) // 2
    bottom = top + crop_height
    left = (orig_width - crop_width) // 2
    right = left + crop_width

    # Check if cropped area is within image boundaries
    if top >= 0 and bottom <= orig_height and left >= 0 and right <= orig_width:
        image = image[..., top:bottom, left:right]
        return image

    # Padding with zeros
    new_height = max(crop_height, orig_height)
    new_width = max(crop_width, orig_width)
    new_shape = image.shape[:-2] + (new_height, new_width)
    new_image = np.zeros_like(image, shape=new_shape, dtype=np.float32)

    top_pad = (new_height - orig_height) // 2
    bottom_pad = top_pad + orig_height
    left_pad = (new_width - orig_width) // 2
    right_pad = left_pad + orig_width
    new_image[..., top_pad:bottom_pad, left_pad:right_pad] = image

    top += top_pad
    bottom += top_pad
    left += left_pad
    right += left_pad

    new_image = new_image[
        ..., max(0, top) : min(new_height, bottom), max(0, left) : min(new_width, right)
    ]

    return new_image


def normalize(
    image: NumpyArray,
    mean: Union[float, list[float]],
    std: Union[float, list[float]],
) -> NumpyArray:
    num_channels = image.shape[1] if len(image.shape) == 4 else image.shape[0]

    if not np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32)

    mean_list = mean if isinstance(mean, list) else [mean] * num_channels

    if len(mean_list) != num_channels:
        raise ValueError(
            f"mean must have the same number of channels as the image, image has {num_channels} channels, got "
            f"{len(mean_list)}"
        )

    mean_arr = np.array(mean_list, dtype=np.float32)

    std_list = std if isinstance(std, list) else [std] * num_channels
    if len(std_list) != num_channels:
        raise ValueError(
            f"std must have the same number of channels as the image, image has {num_channels} channels, got {len(std_list)}"
        )

    std_arr = np.array(std_list, dtype=np.float32)

    image_upd = ((image.T - mean_arr) / std_arr).T
    return image_upd


def resize(
    image: Image.Image,
    size: Union[int, tuple[int, int]],
    resample: Union[int, Image.Resampling] = Image.Resampling.BILINEAR,
) -> Image.Image:
    if isinstance(size, tuple):
        return image.resize(size, resample)

    height, width = image.height, image.width
    short, long = (width, height) if width <= height else (height, width)

    new_short, new_long = size, int(size * long / short)
    if width <= height:
        new_size = (new_short, new_long)
    else:
        new_size = (new_long, new_short)
    return image.resize(new_size, resample)


def rescale(image: NumpyArray, scale: float, dtype: type = np.float32) -> NumpyArray:
    return (image * scale).astype(dtype)


def pil2ndarray(image: Union[Image.Image, NumpyArray]) -> NumpyArray:
    if isinstance(image, Image.Image):
        return np.asarray(image).transpose((2, 0, 1))
    return image


def pad2square(
    image: Image.Image,
    size: int,
    fill_color: Union[str, int, tuple[int, ...]] = 0,
) -> Image.Image:
    height, width = image.height, image.width

    left, right = 0, width
    top, bottom = 0, height

    crop_required = False
    if width > size:
        left = (width - size) // 2
        right = left + size
        crop_required = True

    if height > size:
        top = (height - size) // 2
        bottom = top + size
        crop_required = True

    new_image = Image.new(mode="RGB", size=(size, size), color=fill_color)
    new_image.paste(image.crop((left, top, right, bottom)) if crop_required else image)
    return new_image
