import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, AddedToken
from huggingface_hub import hf_hub_download


def load_tokenizer(repo_id: str, cache_dir: Path, max_length: int = 512) -> Tokenizer:
    config_path = hf_hub_download(
        repo_id=repo_id, filename="config.json", cache_dir=str(cache_dir)
    )

    tokenizer_path = hf_hub_download(
        repo_id=repo_id, filename="tokenizer.json", cache_dir=str(cache_dir)
    )

    tokenizer_config_path = hf_hub_download(
        repo_id=repo_id, filename="tokenizer_config.json", cache_dir=str(cache_dir)
    )

    tokens_map_path = hf_hub_download(
        repo_id=repo_id, filename="special_tokens_map.json", cache_dir=str(cache_dir)
    )

    with open(str(config_path)) as config_file:
        config = json.load(config_file)

    with open(str(tokenizer_config_path)) as tokenizer_config_file:
        tokenizer_config = json.load(tokenizer_config_file)

    with open(str(tokens_map_path)) as tokens_map_file:
        tokens_map = json.load(tokens_map_file)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=min(tokenizer_config["model_max_length"], max_length))
    tokenizer.enable_padding(
        pad_id=config.get("pad_token_id", 0), pad_token=tokenizer_config["pad_token"]
    )

    for token in tokens_map.values():
        if isinstance(token, str):
            tokenizer.add_special_tokens([token])
        elif isinstance(token, dict):
            tokenizer.add_special_tokens([AddedToken(**token)])

    return tokenizer


def normalize(input_array, p=2, dim=1, eps=1e-12) -> np.ndarray:
    # Calculate the Lp norm along the specified dimension
    norm = np.linalg.norm(input_array, ord=p, axis=dim, keepdims=True)
    norm = np.maximum(norm, eps)  # Avoid division by zero
    normalized_array = input_array / norm
    return normalized_array
