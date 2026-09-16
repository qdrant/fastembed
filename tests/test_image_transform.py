from PIL import Image

from fastembed.image.transform.functional import resize
from fastembed.image.transform.operators import Resize


def test_resize_tuple_converts_from_height_width_to_pillow_order():
    """A ``(height, width)`` size must reach Pillow as ``(width, height)``.

    ``Transform.from_config`` builds the tuple as ``(size["height"], size["width"])``,
    so every tuple reaching ``resize`` is in fastembed's height-first order, while
    ``PIL.Image.resize`` takes width first. The two agree for square sizes, which is
    why this went unnoticed.
    """
    image = Image.new("RGB", (300, 300))

    resized = resize(image, size=(100, 200))

    assert resized.size == (200, 100)  # PIL reports (width, height)


def test_resize_operator_produces_requested_height_and_width():
    image = Image.new("RGB", (300, 300))

    resized = Resize(size=(100, 200))([image])[0]

    width, height = resized.size
    assert (height, width) == (100, 200)


def test_resize_square_tuple_is_unchanged():
    """The square case behaved correctly before and must keep doing so."""
    image = Image.new("RGB", (300, 200))

    assert resize(image, size=(224, 224)).size == (224, 224)


def test_resize_int_keeps_shortest_edge_behaviour():
    """The int branch already emitted Pillow order; it must not be disturbed."""
    landscape = Image.new("RGB", (400, 200))
    portrait = Image.new("RGB", (200, 400))

    # size sets the shortest edge, and the aspect ratio is preserved.
    assert resize(landscape, size=100).size == (200, 100)
    assert resize(portrait, size=100).size == (100, 200)
