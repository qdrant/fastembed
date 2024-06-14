from typing import Sized, Tuple, Union

import numpy as np
from PIL import Image


def convert_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image

    image = image.convert("RGB")
    return image


def center_crop(
    image: Union[Image.Image, np.ndarray],
    size: Tuple[int, int],
) -> np.ndarray:
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
    new_image = np.zeros_like(image, shape=new_shape)

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
    image: np.ndarray,
    mean=Union[float, np.ndarray],
    std=Union[float, np.ndarray],
) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a numpy array")

    num_channels = image.shape[1] if len(image.shape) == 4 else image.shape[0]

    if not np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32)

    if isinstance(mean, Sized):
        if len(mean) != num_channels:
            raise ValueError(
                f"mean must have {num_channels} elements if it is an iterable, got {len(mean)}"
            )
    else:
        mean = [mean] * num_channels
    mean = np.array(mean, dtype=image.dtype)

    if isinstance(std, Sized):
        if len(std) != num_channels:
            raise ValueError(
                f"std must have {num_channels} elements if it is an iterable, got {len(std)}"
            )
    else:
        std = [std] * num_channels
    std = np.array(std, dtype=image.dtype)

    image = ((image.T - mean) / std).T
    return image


def resize(
    image: Image,
    size: Union[int, Tuple[int, int]],
    resample: Image.Resampling = Image.Resampling.BILINEAR,
) -> Image:
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


def rescale(image: np.ndarray, scale: float, dtype=np.float32) -> np.ndarray:
    return (image * scale).astype(dtype)


def pil2ndarray(image: Union[Image.Image, np.ndarray]):
    if isinstance(image, Image.Image):
        return np.asarray(image).transpose((2, 0, 1))
    return image
