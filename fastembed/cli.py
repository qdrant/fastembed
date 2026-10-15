import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fastembed")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Serve local text embeddings over HTTP")
    serve.add_argument(
        "--model-name", help="Supported model name, using the library default if omitted"
    )
    serve.add_argument(
        "--model-path", type=Path, help="Local directory containing the model files"
    )
    serve.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=_positive_int, default=8080)
    serve.add_argument("--threads", type=_positive_int, help="ONNX Runtime threads")
    serve.add_argument("--batch-size", type=_positive_int, default=256)
    serve.add_argument("--cache-dir", help="Model cache directory")
    args = parser.parse_args(argv)
    if args.port > 65535:
        parser.error("port must be between 1 and 65535")
    if args.model_path is not None and not args.model_path.is_dir():
        parser.error("model-path must be an existing directory containing the model files")

    try:
        import uvicorn

        from fastembed.server import create_app
    except ImportError as exc:
        parser.exit(
            1, f"Server dependencies are unavailable. Install 'fastembed[server]'.\n{exc}\n"
        )

    from fastembed import TextEmbedding
    from fastembed.common.types import Device

    options: dict[str, Any] = {
        "cuda": Device(args.device),
        "cache_dir": args.cache_dir,
        "threads": args.threads,
    }
    if args.model_name is not None:
        options["model_name"] = args.model_name
    if args.model_path is not None:
        options["specific_model_path"] = str(args.model_path)
    try:
        model = TextEmbedding(**options)
        app = create_app(model, batch_size=args.batch_size)
    except Exception as exc:
        parser.exit(1, f"Unable to load the embedding model.\n{exc}\n")
    uvicorn.run(app, host=args.host, port=args.port)
