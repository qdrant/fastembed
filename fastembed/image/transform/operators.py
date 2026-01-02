from typing import Any
import math

from PIL import Image

from fastembed.common.types import NumpyArray
from fastembed.image.transform.functional import (
    center_crop,
    convert_to_rgb,
    crop_ndarray,
    normalize,
    pil2ndarray,
    rescale,
    resize,
    resize_longest_edge,
    resize_ndarray,
    pad2square,
)


class Transform:
    def __call__(self, images: list[Any]) -> list[Image.Image] | list[NumpyArray]:
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
    def __init__(self, mean: float | list[float], std: float | list[float]):
        self.mean = mean
        self.std = std

    def __call__(  # type: ignore[override]
        self, images: list[NumpyArray] | list[list[NumpyArray]]
    ) -> list[NumpyArray] | list[list[NumpyArray]]:
        if images and isinstance(images[0], list):
            # Nested structure from ImageSplitter
            return [
                [normalize(image, mean=self.mean, std=self.std) for image in img_patches]  # type: ignore[arg-type]
                for img_patches in images
            ]
        else:
            # Flat structure (backward compatibility)
            return [normalize(image, mean=self.mean, std=self.std) for image in images]  # type: ignore[arg-type]


class Resize(Transform):
    def __init__(
        self,
        size: int | tuple[int, int],
        resample: Image.Resampling = Image.Resampling.BICUBIC,
    ):
        self.size = size
        self.resample = resample

    def __call__(self, images: list[Image.Image]) -> list[Image.Image]:
        return [resize(image, size=self.size, resample=self.resample) for image in images]


class Rescale(Transform):
    def __init__(self, scale: float = 1 / 255):
        self.scale = scale

    def __call__(  # type: ignore[override]
        self, images: list[NumpyArray] | list[list[NumpyArray]]
    ) -> list[NumpyArray] | list[list[NumpyArray]]:
        if images and isinstance(images[0], list):
            # Nested structure from ImageSplitter
            return [
                [rescale(image, scale=self.scale) for image in img_patches]  # type: ignore[arg-type]
                for img_patches in images
            ]
        else:
            # Flat structure (backward compatibility)
            return [rescale(image, scale=self.scale) for image in images]  # type: ignore[arg-type]


class PILtoNDarray(Transform):
    def __call__(self, images: list[Image.Image | NumpyArray]) -> list[NumpyArray]:
        return [pil2ndarray(image) for image in images]


class PadtoSquare(Transform):
    def __init__(
        self,
        size: int,
        fill_color: str | int | tuple[int, ...],
    ):
        self.size = size
        self.fill_color = fill_color

    def __call__(self, images: list[Image.Image]) -> list[Image.Image]:
        return [
            pad2square(image=image, size=self.size, fill_color=self.fill_color) for image in images
        ]


class ResizeLongestEdge(Transform):
    """Resize images so the longest edge equals target size, preserving aspect ratio."""

    def __init__(
        self,
        size: int,
        resample: Image.Resampling = Image.Resampling.LANCZOS,
    ):
        self.size = size
        self.resample = resample

    def __call__(self, images: list[Image.Image]) -> list[Image.Image]:
        return [resize_longest_edge(image, self.size, self.resample) for image in images]


class ResizeForVisionEncoder(Transform):
    """
    Resize both dimensions to be multiples of vision_encoder_max_size.
    Preserves aspect ratio approximately.
    Works on numpy arrays in (C, H, W) format.
    """

    def __init__(
        self,
        max_size: int,
        resample: Image.Resampling = Image.Resampling.LANCZOS,
    ):
        self.max_size = max_size
        self.resample = resample

    def __call__(self, images: list[NumpyArray]) -> list[NumpyArray]:
        result = []
        for image in images:
            # Assume (C, H, W) format
            _, height, width = image.shape

            aspect_ratio = width / height

            if width >= height:
                # Calculate new width as multiple of max_size
                new_width = math.ceil(width / self.max_size) * self.max_size
                new_height = int(new_width / aspect_ratio)
                new_height = math.ceil(new_height / self.max_size) * self.max_size
            else:
                # Calculate new height as multiple of max_size
                new_height = math.ceil(height / self.max_size) * self.max_size
                new_width = int(new_height * aspect_ratio)
                new_width = math.ceil(new_width / self.max_size) * self.max_size

            # Resize using the ndarray resize function
            resized = resize_ndarray(
                image,
                size=(new_width, new_height),  # PIL expects (width, height)
                resample=self.resample,
                channel_first=True,
            )
            result.append(resized)

        return result


class ImageSplitter(Transform):
    """
    Split images into grid of patches plus a global view.

    If image dimensions exceed max_size:
    - Divide into ceil(H/max_size) x ceil(W/max_size) patches
    - Each patch is cropped from the image
    - Add a global view (original resized to max_size x max_size)

    If image is smaller than max_size:
    - Return single image unchanged

    Works on numpy arrays in (C, H, W) format.
    """

    def __init__(
        self,
        max_size: int,
        resample: Image.Resampling = Image.Resampling.LANCZOS,
    ):
        self.max_size = max_size
        self.resample = resample

    def __call__(self, images: list[NumpyArray]) -> list[list[NumpyArray]]:  # type: ignore[override]
        result = []

        for image in images:
            # Assume (C, H, W) format
            _, height, width = image.shape
            max_height = max_width = self.max_size

            frames = []

            if height > max_height or width > max_width:
                # Calculate the number of splits needed
                num_splits_h = math.ceil(height / max_height)
                num_splits_w = math.ceil(width / max_width)

                # Calculate optimal patch dimensions
                optimal_height = math.ceil(height / num_splits_h)
                optimal_width = math.ceil(width / num_splits_w)

                # Generate patches in grid order (row by row)
                for r in range(num_splits_h):
                    for c in range(num_splits_w):
                        # Calculate crop coordinates
                        start_x = c * optimal_width
                        start_y = r * optimal_height
                        end_x = min(start_x + optimal_width, width)
                        end_y = min(start_y + optimal_height, height)

                        # Crop the patch
                        cropped = crop_ndarray(
                            image, x1=start_x, y1=start_y, x2=end_x, y2=end_y, channel_first=True
                        )
                        frames.append(cropped)

                # Add global view (resized to max_size x max_size)
                global_view = resize_ndarray(
                    image,
                    size=(max_width, max_height),  # PIL expects (width, height)
                    resample=self.resample,
                    channel_first=True,
                )
                frames.append(global_view)
            else:
                # Image is small enough, no splitting needed
                frames.append(image)

            # Append (not extend) to preserve per-image grouping
            result.append(frames)

        return result


class SquareResize(Transform):
    """
    Resize images to square dimensions (max_size x max_size).
    Works on numpy arrays in (C, H, W) format.
    """

    def __init__(
        self,
        size: int,
        resample: Image.Resampling = Image.Resampling.LANCZOS,
    ):
        self.size = size
        self.resample = resample

    def __call__(self, images: list[NumpyArray]) -> list[list[NumpyArray]]:  # type: ignore[override]
        return [
            [
                resize_ndarray(
                    image, size=(self.size, self.size), resample=self.resample, channel_first=True
                )
            ]
            for image in images
        ]


class Compose:
    def __init__(self, transforms: list[Transform]):
        self.transforms = transforms

    def __call__(
        self, images: list[Image.Image] | list[NumpyArray]
    ) -> list[NumpyArray] | list[Image.Image]:
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
                    - {"longest_edge"}

        Returns:
            Compose: Image processor.
        """
        transforms: list[Transform] = []
        cls._get_convert_to_rgb(transforms, config)
        cls._get_resize(transforms, config)
        cls._get_pad2square(transforms, config)
        cls._get_center_crop(transforms, config)
        cls._get_pil2ndarray(transforms, config)
        cls._get_image_splitting(transforms, config)
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
        elif mode == "Idefics3ImageProcessor":
            if config.get("do_resize", False):
                size = config.get("size", {})
                if "longest_edge" not in size:
                    raise ValueError(
                        "Size dictionary must contain 'longest_edge' key for Idefics3ImageProcessor"
                    )

                # Handle resample parameter - can be int enum or PIL.Image.Resampling
                resample = config.get("resample", Image.Resampling.LANCZOS)
                if isinstance(resample, int):
                    resample = Image.Resampling(resample)

                transforms.append(
                    ResizeLongestEdge(
                        size=size["longest_edge"],
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
        elif mode == "Idefics3ImageProcessor":
            pass
        else:
            raise ValueError(f"Preprocessor {mode} is not supported")

    @staticmethod
    def _get_pil2ndarray(transforms: list[Transform], config: dict[str, Any]) -> None:
        transforms.append(PILtoNDarray())

    @classmethod
    def _get_image_splitting(cls, transforms: list[Transform], config: dict[str, Any]) -> None:
        """
        Add image splitting transforms for Idefics3.
        Handles conditional logic: splitting vs square resize.
        Must be called AFTER PILtoNDarray.
        """
        mode = config.get("image_processor_type", "CLIPImageProcessor")

        if mode == "Idefics3ImageProcessor":
            do_splitting = config.get("do_image_splitting", False)
            max_size = config.get("max_image_size", {}).get("longest_edge", 512)
            resample = config.get("resample", Image.Resampling.LANCZOS)
            if isinstance(resample, int):
                resample = Image.Resampling(resample)

            if do_splitting:
                transforms.append(ResizeForVisionEncoder(max_size, resample))
                transforms.append(ImageSplitter(max_size, resample))
            else:
                transforms.append(SquareResize(max_size, resample))

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
    def _interpolation_resolver(resample: str | None = None) -> Image.Resampling:
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
