from typing import Any, Union, Optional

from PIL import Image

from fastembed.common.types import NumpyArray
from fastembed.image.transform.functional import (
    center_crop,
    convert_to_rgb,
    normalize,
    pil2ndarray,
    rescale,
    resize,
    pad2square,
)


class Transform:
    def __call__(self, images: list[Any]) -> Union[list[Image.Image], list[NumpyArray]]:
        raise NotImplementedError("Subclasses must implement this method")


class ConvertToRGB(Transform):
    def __call__(self, images: list[Image.Image]) -> list[Image.Image]:
        return [convert_to_rgb(image=image) for image in images]


class CenterCrop(Transform):
    def __init__(self, size: tuple[int, int]):
        self.size = size

    def __call__(self, images: list[Image.Image]) -> list[NumpyArray]:
        return [center_crop(image=image, size=self.size) for image in images]


class Normalize(Transform):
    def __init__(self, mean: Union[float, list[float]], std: Union[float, list[float]]):
        self.mean = mean
        self.std = std

    def __call__(self, images: list[NumpyArray]) -> list[NumpyArray]:
        return [normalize(image, mean=self.mean, std=self.std) for image in images]


class Resize(Transform):
    def __init__(
        self,
        size: Union[int, tuple[int, int]],
        resample: Image.Resampling = Image.Resampling.BICUBIC,
    ):
        self.size = size
        self.resample = resample

    def __call__(self, images: list[Image.Image]) -> list[Image.Image]:
        return [resize(image, size=self.size, resample=self.resample) for image in images]


class Rescale(Transform):
    def __init__(self, scale: float = 1 / 255):
        self.scale = scale

    def __call__(self, images: list[NumpyArray]) -> list[NumpyArray]:
        return [rescale(image, scale=self.scale) for image in images]


class PILtoNDarray(Transform):
    def __call__(self, images: list[Union[Image.Image, NumpyArray]]) -> list[NumpyArray]:
        return [pil2ndarray(image) for image in images]


class PadtoSquare(Transform):
    def __init__(
        self,
        size: int,
        fill_color: Union[str, int, tuple[int, ...]],
    ):
        self.size = size
        self.fill_color = fill_color

    def __call__(self, images: list[Image.Image]) -> list[Image.Image]:
        return [
            pad2square(image=image, size=self.size, fill_color=self.fill_color) for image in images
        ]


class Compose:
    def __init__(self, transforms: list[Transform]):
        self.transforms = transforms

    def __call__(
        self, images: Union[list[Image.Image], list[NumpyArray]]
    ) -> Union[list[NumpyArray], list[Image.Image]]:
        for transform in self.transforms:
            images = transform(images)
        return images

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Compose":
        """Creates processor from a config dict.
        Args:
            config (dict[str, Any]): Configuration dictionary.

                Valid keys:
                    - do_resize
                    - resize_mode
                    - size
                    - fill_color
                    - do_center_crop
                    - crop_size
                    - do_rescale
                    - rescale_factor
                    - do_normalize
                    - image_mean
                    - mean
                    - image_std
                    - std
                    - resample
                    - interpolation
                Valid size keys (nested):
                    - {"height", "width"}
                    - {"shortest_edge"}

        Returns:
            Compose: Image processor.
        """
        transforms: list[Transform] = []
        cls._get_convert_to_rgb(transforms, config)
        cls._get_resize(transforms, config)
        cls._get_pad2square(transforms, config)
        cls._get_center_crop(transforms, config)
        cls._get_pil2ndarray(transforms, config)
        cls._get_rescale(transforms, config)
        cls._get_normalize(transforms, config)
        return cls(transforms=transforms)

    @staticmethod
    def _get_convert_to_rgb(transforms: list[Transform], config: dict[str, Any]) -> None:
        transforms.append(ConvertToRGB())

    @classmethod
    def _get_resize(cls, transforms: list[Transform], config: dict[str, Any]) -> None:
        mode = config.get("image_processor_type", "CLIPImageProcessor")
        if mode in ("CLIPImageProcessor", "SiglipImageProcessor"):
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
        elif mode == "JinaCLIPImageProcessor":
            interpolation = config.get("interpolation")
            if isinstance(interpolation, str):
                resample = cls._interpolation_resolver(interpolation)
            else:
                resample = interpolation or Image.Resampling.BICUBIC

            if "size" in config:
                resize_mode = config.get("resize_mode", "shortest")
                if resize_mode == "shortest":
                    transforms.append(
                        Resize(
                            size=config["size"],
                            resample=resample,
                        )
                    )
        else:
            raise ValueError(f"Preprocessor {mode} is not supported")

    @staticmethod
    def _get_center_crop(transforms: list[Transform], config: dict[str, Any]) -> None:
        mode = config.get("image_processor_type", "CLIPImageProcessor")
        if mode in ("CLIPImageProcessor", "SiglipImageProcessor"):
            if config.get("do_center_crop", False):
                crop_size_raw = config["crop_size"]
                crop_size: tuple[int, int]
                if isinstance(crop_size_raw, int):
                    crop_size = (crop_size_raw, crop_size_raw)
                elif isinstance(crop_size_raw, dict):
                    crop_size = (crop_size_raw["height"], crop_size_raw["width"])
                else:
                    raise ValueError(f"Invalid crop size: {crop_size_raw}")
                transforms.append(CenterCrop(size=crop_size))
        elif mode == "ConvNextFeatureExtractor":
            pass
        elif mode == "JinaCLIPImageProcessor":
            pass
        else:
            raise ValueError(f"Preprocessor {mode} is not supported")

    @staticmethod
    def _get_pil2ndarray(transforms: list[Transform], config: dict[str, Any]) -> None:
        transforms.append(PILtoNDarray())

    @staticmethod
    def _get_rescale(transforms: list[Transform], config: dict[str, Any]) -> None:
        if config.get("do_rescale", True):
            rescale_factor = config.get("rescale_factor", 1 / 255)
            transforms.append(Rescale(scale=rescale_factor))

    @staticmethod
    def _get_normalize(transforms: list[Transform], config: dict[str, Any]) -> None:
        if config.get("do_normalize", False):
            transforms.append(Normalize(mean=config["image_mean"], std=config["image_std"]))
        elif "mean" in config and "std" in config:
            transforms.append(Normalize(mean=config["mean"], std=config["std"]))

    @staticmethod
    def _get_pad2square(transforms: list[Transform], config: dict[str, Any]) -> None:
        mode = config.get("image_processor_type", "CLIPImageProcessor")
        if mode == "CLIPImageProcessor":
            pass
        elif mode == "ConvNextFeatureExtractor":
            pass
        elif mode == "JinaCLIPImageProcessor":
            transforms.append(
                PadtoSquare(
                    size=config["size"],
                    fill_color=config.get("fill_color", 0),
                )
            )

    @staticmethod
    def _interpolation_resolver(resample: Optional[str] = None) -> Image.Resampling:
        interpolation_map = {
            "nearest": Image.Resampling.NEAREST,
            "lanczos": Image.Resampling.LANCZOS,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "box": Image.Resampling.BOX,
            "hamming": Image.Resampling.HAMMING,
        }

        if resample and (method := interpolation_map.get(resample.lower())):
            return method

        raise ValueError(f"Unknown interpolation method: {resample}")
