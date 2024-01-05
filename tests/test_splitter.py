import os
from pathlib import Path

import pytest

from fastembed.embedding import DefaultEmbedding
from fastembed.types import TextSplitterConfig


@pytest.mark.parametrize(["chunk_size", "chunk_overlap"], [[500, 50], [1000, 100]])
def test_embedding(chunk_size: int, chunk_overlap: int):
    is_ubuntu_ci = os.getenv("IS_UBUNTU_CI")

    for model_desc in DefaultEmbedding.list_supported_models():
        if is_ubuntu_ci == "false" and model_desc["size_in_GB"] > 1:
            continue

        p = Path(__file__).with_name("state_of_the_union.txt")
        with open(p, encoding='utf-8') as f:
            text = f.read()
            embedding = DefaultEmbedding(
                model_name=model_desc["model"],
                splitter_config=TextSplitterConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            )
            texts = embedding.split_text(text)
            for text in texts:
                assert len(embedding.model.tokenizer.encode(text)) <= chunk_size
