from typing import List, Tuple, Union, Any, Dict

import numpy as np
from PIL import Image

from fastembed.image.transform.functional import (
    center_crop,
    normalize,
    resize,
    convert_to_rgb,
    rescale,
)


class Transform:
    def __call__(self, images: List) -> Union[List[Image.Image], List[np.ndarray]]:
        raise NotImplementedError("Subclasses must implement this method")


class ConvertToRGB(Transform):
    def __call__(self, images: List[Image.Image]) -> List[Image.Image]:
        return [convert_to_rgb(image=image) for image in images]


class CenterCrop(Transform):
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, images: List[Image.Image]) -> List[np.ndarray]:
        return [center_crop(image=image, size=self.size) for image in images]


class Normalize(Transform):
    def __init__(self, mean: Union[float, List[float]], std: Union[float, List[float]]):
        self.mean = mean
        self.std = std

    def __call__(self, images: List[np.ndarray]) -> List[np.ndarray]:
        return [normalize(image, mean=self.mean, std=self.std) for image in images]


class Resize(Transform):
    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        resample: Image.Resampling = Image.Resampling.BICUBIC,
    ):
        self.size = size
        self.resample = resample

    def __call__(self, images: List[Image.Image]) -> List[Image.Image]:
        return [resize(image, size=self.size, resample=self.resample) for image in images]


class Rescale(Transform):
    def __init__(self, scale: float = 1 / 255):
        self.scale = scale

    def __call__(self, images: List[np.ndarray]) -> List[np.ndarray]:
        return [rescale(image, scale=self.scale) for image in images]


class Compose:
    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def __call__(
        self, images: Union[List[Image.Image], List[np.ndarray]]
    ) -> Union[List[np.ndarray], List[Image.Image]]:
        for transform in self.transforms:
            images = transform(images)
        return images

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Compose":
        transforms = [ConvertToRGB()]
        if config.get("do_resize", False):
            size = config["size"]
            if "shortest_edge" in size:
                size = size["shortest_edge"]
            elif "height" in size and "width" in size:
                size = (size["height"], size["width"])
            else:
                raise ValueError(
                    "Size must contain either 'shortest_edge' or 'height' and 'width'."
                )
            transforms.append(
                Resize(size=size, resample=config.get("resample", Image.Resampling.BICUBIC))
            )
        if config.get("do_center_crop", False):
            crop_size = config["crop_size"]
            if isinstance(crop_size, int):
                crop_size = (crop_size, crop_size)
            elif isinstance(crop_size, dict):
                crop_size = (crop_size["height"], crop_size["width"])
            else:
                raise ValueError(f"Invalid crop size: {crop_size}")
            transforms.append(CenterCrop(size=crop_size))
        if config.get("do_rescale", True):
            rescale_factor = config.get("rescale_factor", 1 / 255)
            transforms.append(Rescale(scale=rescale_factor))
        if config.get("do_normalize", False):
            transforms.append(Normalize(mean=config["image_mean"], std=config["image_std"]))
        return cls(transforms=transforms)
