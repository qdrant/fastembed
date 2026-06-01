#!/usr/bin/env python3
"""Generate LateOn canonical test values from the PyLate reference implementation.

This script prints the abridged document and query vectors used by
``tests/test_late_interaction_embeddings.py``. It intentionally uses PyLate,
not FastEmbed, so the generated values come from the original reference model.

Example:
    python scripts/generate_lateon_canonical_values.py
"""

from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np


DEFAULT_MODEL = "lightonai/LateOn"
DEFAULT_TEXT = "Hello World"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate canonical LateOn test vectors with PyLate."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="PyLate/HF model name")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text to encode")
    parser.add_argument(
        "--rows",
        type=int,
        default=5,
        help="Number of token rows to print from each embedding",
    )
    parser.add_argument(
        "--dims",
        type=int,
        default=5,
        help="Number of dimensions to print from each token embedding",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=5,
        help="Decimal precision for printed values",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device passed to PyLate encode, e.g. cpu or cuda",
    )
    return parser.parse_args()


def load_pylate_model(model_name: str):
    try:
        from pylate import models
    except ImportError as exc:
        raise SystemExit(
            "PyLate is required to generate reference values. "
            "Install it with `pip install pylate`."
        ) from exc

    return models.ColBERT(model_name_or_path=model_name)


def encode_reference(model, texts: Sequence[str], *, is_query: bool, device: str) -> np.ndarray:
    embeddings = model.encode(
        list(texts),
        batch_size=1,
        is_query=is_query,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device,
    )
    return np.asarray(embeddings[0], dtype=np.float32)


def format_dict_entry(model_name: str, values: np.ndarray, *, precision: int) -> str:
    array = np.array2string(
        values,
        precision=precision,
        separator=", ",
        suppress_small=False,
    )
    # Indent nested array lines to match the style in tests/test_late_interaction_embeddings.py.
    array = "\n".join(f"        {line}" for line in array.splitlines())
    return f'    "{model_name}": np.array(\n{array}\n    ),'


def main() -> None:
    args = parse_args()
    model = load_pylate_model(args.model)

    document_embedding = encode_reference(model, [args.text], is_query=False, device=args.device)
    query_embedding = encode_reference(model, [args.text], is_query=True, device=args.device)

    document_values = document_embedding[: args.rows, : args.dims]
    query_values = query_embedding[: args.rows, : args.dims]

    print("# Generated from the PyLate reference implementation, not FastEmbed.")
    print("#")
    print("# Reference code:")
    print("#   from pylate import models")
    print(f"#   model = models.ColBERT(model_name_or_path={args.model!r})")
    print(
        "#   model.encode([text], is_query=False/True, "
        "convert_to_numpy=True, normalize_embeddings=True)"
    )
    print(f"# text = {args.text!r}")
    print(f"# document_shape = {tuple(document_embedding.shape)}")
    print(f"# query_shape = {tuple(query_embedding.shape)}")
    print()
    print("# CANONICAL_COLUMN_VALUES entry")
    print(format_dict_entry(args.model, document_values, precision=args.precision))
    print()
    print("# CANONICAL_QUERY_VALUES entry")
    print(format_dict_entry(args.model, query_values, precision=args.precision))


if __name__ == "__main__":
    main()
