import pytest
from PIL import Image

from fastembed.image.transform.functional import resize


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
