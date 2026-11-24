"""Reference checks for the larger Chinese BGE model.

Canonical values come from the original PyTorch model using the following snippet.

    from transformers import AutoModel, AutoTokenizer
    import torch

    name = "BAAI/bge-large-zh-v1.5"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).eval()
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        vectors = torch.nn.functional.normalize(
            model(**inputs).last_hidden_state[:, 0], p=2, dim=1
        )
    print(vectors[:, :8].tolist())
"""

import os

import numpy as np
import pytest

from fastembed import TextEmbedding
from tests.utils import delete_model_cache, is_manual_run, should_test_model

MODEL_NAME = "BAAI/bge-large-zh-v1.5"
REFERENCE_CASES = [
    (
        "hello world",
        [
            -0.025394167751073837,
            -0.0016842425102367997,
            -0.094914011657238,
            -0.007070891559123993,
            0.0027618478052318096,
            0.050447918474674225,
            -0.0512826070189476,
            0.021190207451581955,
        ],
    ),
    (
        "你好世界",
        [
            0.04448169842362404,
            -0.023043543100357056,
            -0.06824296712875366,
            -0.005314783658832312,
            -0.025446804240345955,
            0.016195841133594513,
            -0.06172054260969162,
            0.028417708352208138,
        ],
    ),
    (
        "北京是中国的首都。",
        [
            -0.07171106338500977,
            0.028986480087041855,
            0.0031650469172745943,
            0.02360720932483673,
            0.00862447265535593,
            0.05719650164246559,
            -0.0070158555172383785,
            -0.037890322506427765,
        ],
    ),
    (
        "为这个句子生成表示以用于检索相关文章： 中国的首都是哪里？",
        [
            -0.06612120568752289,
            0.03332896530628204,
            0.028995418921113014,
            0.03479088470339775,
            0.005401355680078268,
            0.07183656841516495,
            0.001583364442922175,
            -0.029327189549803734,
        ],
    ),
]


@pytest.fixture(scope="module")
def model():
    description = next(
        item for item in TextEmbedding._list_supported_models() if item.model == MODEL_NAME
    )
    if not should_test_model(description, "", os.getenv("CI"), is_manual_run()):
        pytest.skip("The large-model check runs in the scheduled model matrix")
    embedding = TextEmbedding(MODEL_NAME, threads=2, cuda=False)
    yield embedding
    if os.getenv("CI"):
        delete_model_cache(embedding.model._model_dir)


@pytest.mark.parametrize("text,expected", REFERENCE_CASES)
def test_reference_embeddings(model, text, expected):
    passage = next(iter(model.passage_embed([text])))
    query = next(iter(model.query_embed(text)))
    assert passage.shape == query.shape == (1024,)
    assert np.allclose(passage[:8], expected, atol=1e-3)
    assert np.allclose(query[:8], expected, atol=1e-3)
    assert np.linalg.norm(passage) == pytest.approx(1.0, abs=1e-6)


def test_mixed_length_parallel_embeddings(model):
    texts = [text for text, _ in REFERENCE_CASES]
    sequential = np.array(list(model.embed(texts, batch_size=4)))
    parallel = np.array(list(model.embed(texts, batch_size=1, parallel=2)))
    assert sequential.shape == parallel.shape == (len(texts), 1024)
    assert np.allclose(sequential, parallel, atol=1e-5)


def test_long_chinese_input_is_truncated(model):
    assert model.token_count("中国" * 600) == 512
