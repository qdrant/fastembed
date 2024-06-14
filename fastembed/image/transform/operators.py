from typing import Any, Dict, List, Tuple, Union

import numpy as np
from PIL import Image

from fastembed.image.transform.functional import (
    center_crop,
    convert_to_rgb,
    normalize,
    pil2ndarray,
    rescale,
    resize,
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
        return [
            resize(image, size=self.size, resample=self.resample) for image in images
        ]


class Rescale(Transform):
    def __init__(self, scale: float = 1 / 255):
        self.scale = scale

    def __call__(self, images: List[np.ndarray]) -> List[np.ndarray]:
        return [rescale(image, scale=self.scale) for image in images]


class PILtoNDarray(Transform):
    def __call__(
        self, images: List[Union[Image.Image, np.ndarray]]
    ) -> List[np.ndarray]:
        return [pil2ndarray(image) for image in images]


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
        """Creates processor from a config dict.
        Args:
            config (Dict[str, Any]): Configuration dictionary.

                Valid keys:
                    - do_resize
                    - size
                    - do_center_crop
                    - crop_size
                    - do_rescale
                    - rescale_factor
                    - do_normalize
                    - image_mean
                    - image_std
                Valid size keys (nested):
                    - {"height", "width"}
                    - {"shortest_edge"}

        Returns:
            Compose: Image processor.
        """
        transforms = []
        cls._get_convert_to_rgb(transforms, config)
        cls._get_resize(transforms, config)
        cls._get_center_crop(transforms, config)
        cls._get_pil2ndarray(transforms, config)
        cls._get_rescale(transforms, config)
        cls._get_normalize(transforms, config)
        return cls(transforms=transforms)

    @staticmethod
    def _get_convert_to_rgb(transforms: List[Transform], config: Dict[str, Any]):
        transforms.append(ConvertToRGB())

    @staticmethod
    def _get_resize(transforms: List[Transform], config: Dict[str, Any]):
        mode = config.get("image_processor_type", "CLIPImageProcessor")
        if mode == "CLIPImageProcessor":
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
                    Resize(
                        size=size,
                        resample=config.get("resample", Image.Resampling.BICUBIC),
                    )
                )
        elif mode == "ConvNextFeatureExtractor":
            if "size" in config and "shortest_edge" not in config["size"]:
                raise ValueError(
                    f"Size dictionary must contain 'shortest_edge' key. Got {config['size'].keys()}"
                )
            shortest_edge = config["size"]["shortest_edge"]
            crop_pct = config.get("crop_pct", 0.875)
            if shortest_edge < 384:
                # maintain same ratio, resizing shortest edge to shortest_edge/crop_pct
                resize_shortest_edge = int(shortest_edge / crop_pct)
                transforms.append(
                    Resize(
                        size=resize_shortest_edge,
                        resample=config.get("resample", Image.Resampling.BICUBIC),
                    )
                )
                transforms.append(CenterCrop(size=(shortest_edge, shortest_edge)))
            else:
                transforms.append(
                    Resize(
                        size=(shortest_edge, shortest_edge),
                        resample=config.get("resample", Image.Resampling.BICUBIC),
                    )
                )

    @staticmethod
    def _get_center_crop(transforms: List[Transform], config: Dict[str, Any]):
        mode = config.get("image_processor_type", "CLIPImageProcessor")
        if mode == "CLIPImageProcessor":
            if config.get("do_center_crop", False):
                crop_size = config["crop_size"]
                if isinstance(crop_size, int):
                    crop_size = (crop_size, crop_size)
                elif isinstance(crop_size, dict):
                    crop_size = (crop_size["height"], crop_size["width"])
                else:
                    raise ValueError(f"Invalid crop size: {crop_size}")
                transforms.append(CenterCrop(size=crop_size))
        elif mode == "ConvNextFeatureExtractor":
            pass
        else:
            raise ValueError(f"Preprocessor {mode} is not supported")

    @staticmethod
    def _get_pil2ndarray(transforms: List[Transform], config: Dict[str, Any]):
        transforms.append(PILtoNDarray())

    @staticmethod
    def _get_rescale(transforms: List[Transform], config: Dict[str, Any]):
        if config.get("do_rescale", True):
            rescale_factor = config.get("rescale_factor", 1 / 255)
            transforms.append(Rescale(scale=rescale_factor))

    @staticmethod
    def _get_normalize(transforms: List[Transform], config: Dict[str, Any]):
        if config.get("do_normalize", False):
            transforms.append(
                Normalize(mean=config["image_mean"], std=config["image_std"])
            )
