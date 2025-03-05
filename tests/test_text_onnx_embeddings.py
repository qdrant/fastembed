import os
import platform

import numpy as np
import pytest

from fastembed.text.text_embedding import TextEmbedding
from tests.utils import delete_model_cache, should_test_model

CANONICAL_VECTOR_VALUES = {
    "BAAI/bge-small-en": np.array([-0.0232, -0.0255, 0.0174, -0.0639, -0.0006]),
    "BAAI/bge-small-en-v1.5": np.array(
        [0.01522374, -0.02271799, 0.00860278, -0.07424029, 0.00386434]
    ),
    "BAAI/bge-small-en-v1.5-quantized": np.array(
        [0.01522374, -0.02271799, 0.00860278, -0.07424029, 0.00386434]
    ),
    "BAAI/bge-small-zh-v1.5": np.array(
        [-0.01023294, 0.07634465, 0.0691722, -0.04458365, -0.03160762]
    ),
    "BAAI/bge-base-en": np.array([0.0115, 0.0372, 0.0295, 0.0121, 0.0346]),
    "BAAI/bge-base-en-v1.5": np.array(
        [0.01129394, 0.05493144, 0.02615099, 0.00328772, 0.02996045]
    ),
    "BAAI/bge-large-en-v1.5": np.array(
        [0.03434538, 0.03316108, 0.02191251, -0.03713358, -0.01577825]
    ),
    "BAAI/bge-large-en-v1.5-quantized": np.array(
        [0.03434538, 0.03316108, 0.02191251, -0.03713358, -0.01577825]
    ),
    "sentence-transformers/all-MiniLM-L6-v2": np.array(
        [-0.034478, 0.03102, 0.00673, 0.02611, -0.039362]
    ),
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": np.array(
        [0.0361, 0.1862, 0.2776, 0.2461, -0.1904]
    ),
    "intfloat/multilingual-e5-large": np.array([0.4544, -0.0968, 0.1054, -1.3753, 0.1500]),
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": np.array(
        [0.0047, 0.1334, -0.0102, 0.0714, 0.1930]
    ),
    "jinaai/jina-embeddings-v2-small-en": np.array([-0.0455, -0.0428, -0.0122, 0.0613, 0.0015]),
    "jinaai/jina-embeddings-v2-base-en": np.array([-0.0332, -0.0509, 0.0287, -0.0043, -0.0077]),
    "jinaai/jina-embeddings-v2-base-de": np.array([-0.0085, 0.0417, 0.0342, 0.0309, -0.0149]),
    "jinaai/jina-embeddings-v2-base-code": np.array([0.0145, -0.0164, 0.0136, -0.0170, 0.0734]),
    "jinaai/jina-embeddings-v2-base-zh": np.array([0.0381, 0.0286, -0.0231, 0.0052, -0.0151]),
    "jinaai/jina-embeddings-v2-base-es": np.array([-0.0108, -0.0092, -0.0373, 0.0171, -0.0301]),
    "nomic-ai/nomic-embed-text-v1": np.array([0.3708, 0.2031, -0.3406, -0.2114, -0.3230]),
    "nomic-ai/nomic-embed-text-v1.5": np.array(
        [-0.15407836, -0.03053198, -3.9138033, 0.1910364, 0.13224715]
    ),
    "nomic-ai/nomic-embed-text-v1.5-Q": np.array(
        [0.0802303, 0.3700881, -4.3053818, 0.4431803, -0.271572]
    ),
    "thenlper/gte-large": np.array(
        [-0.00986551, -0.00018734, 0.00605892, -0.03289612, -0.0387564],
    ),
    "mixedbread-ai/mxbai-embed-large-v1": np.array(
        [0.02295546, 0.03196154, 0.016512, -0.04031524, -0.0219634]
    ),
    "snowflake/snowflake-arctic-embed-xs": np.array([0.0092, 0.0619, 0.0196, 0.009, -0.0114]),
    "snowflake/snowflake-arctic-embed-s": np.array([-0.0416, -0.0867, 0.0209, 0.0554, -0.0272]),
    "snowflake/snowflake-arctic-embed-m": np.array([-0.0329, 0.0364, 0.0481, 0.0016, 0.0328]),
    "snowflake/snowflake-arctic-embed-m-long": np.array(
        [0.0080, -0.0266, -0.0335, 0.0282, 0.0143]
    ),
    "snowflake/snowflake-arctic-embed-l": np.array([0.0189, -0.0673, 0.0183, 0.0124, 0.0146]),
    "Qdrant/clip-ViT-B-32-text": np.array([0.0083, 0.0103, -0.0138, 0.0199, -0.0069]),
    "thenlper/gte-base": np.array([0.0038, 0.0355, 0.0181, 0.0092, 0.0654]),
    "jinaai/jina-clip-v1": np.array([-0.0862, -0.0101, -0.0056, 0.0375, -0.0472]),
}

MULTI_TASK_MODELS = ["jinaai/jina-embeddings-v3"]


@pytest.mark.parametrize("model_name", ["BAAI/bge-small-en-v1.5"])
def test_embedding(model_name: str) -> None:
    is_ci = os.getenv("CI")
    is_mac = platform.system() == "Darwin"
    is_manual = os.getenv("GITHUB_EVENT_NAME") == "workflow_dispatch"

    for model_desc in TextEmbedding._list_supported_models():
        if model_desc.model in MULTI_TASK_MODELS or (
            is_mac and model_desc.model == "nomic-ai/nomic-embed-text-v1.5-Q"
        ):
            continue
        if not should_test_model(model_desc, model_name, is_ci, is_manual):
            continue

        dim = model_desc.dim

        model = TextEmbedding(model_name=model_desc.model)
        docs = ["hello world", "flag embedding"]
        embeddings = list(model.embed(docs))
        embeddings = np.stack(embeddings, axis=0)
        assert embeddings.shape == (2, dim)

        canonical_vector = CANONICAL_VECTOR_VALUES[model_desc.model]
        assert np.allclose(
            embeddings[0, : canonical_vector.shape[0]], canonical_vector, atol=1e-3
        ), model_desc.model
        if is_ci:
            delete_model_cache(model.model._model_dir)


@pytest.mark.parametrize("n_dims,model_name", [(384, "BAAI/bge-small-en-v1.5")])
def test_batch_embedding(n_dims: int, model_name: str) -> None:
    is_ci = os.getenv("CI")
    model = TextEmbedding(model_name=model_name)

    docs = ["hello world", "flag embedding"] * 100
    embeddings = list(model.embed(docs, batch_size=10))
    embeddings = np.stack(embeddings, axis=0)

    assert embeddings.shape == (len(docs), n_dims)
    if is_ci:
        delete_model_cache(model.model._model_dir)


@pytest.mark.parametrize("n_dims,model_name", [(384, "BAAI/bge-small-en-v1.5")])
def test_parallel_processing(n_dims: int, model_name: str) -> None:
    is_ci = os.getenv("CI")
    model = TextEmbedding(model_name=model_name)

    docs = ["hello world", "flag embedding"] * 100
    embeddings = list(model.embed(docs, batch_size=10, parallel=2))
    embeddings = np.stack(embeddings, axis=0)

    embeddings_2 = list(model.embed(docs, batch_size=10, parallel=None))
    embeddings_2 = np.stack(embeddings_2, axis=0)

    embeddings_3 = list(model.embed(docs, batch_size=10, parallel=0))
    embeddings_3 = np.stack(embeddings_3, axis=0)

    assert embeddings.shape == (len(docs), n_dims)
    assert np.allclose(embeddings, embeddings_2, atol=1e-3)
    assert np.allclose(embeddings, embeddings_3, atol=1e-3)

    if is_ci:
        delete_model_cache(model.model._model_dir)


@pytest.mark.parametrize("model_name", ["BAAI/bge-small-en-v1.5"])
def test_lazy_load(model_name: str) -> None:
    is_ci = os.getenv("CI")
    model = TextEmbedding(model_name=model_name, lazy_load=True)
    assert not hasattr(model.model, "model")
    docs = ["hello world", "flag embedding"]
    list(model.embed(docs))
    assert hasattr(model.model, "model")

    model = TextEmbedding(model_name=model_name, lazy_load=True)
    list(model.query_embed(docs))

    model = TextEmbedding(model_name=model_name, lazy_load=True)
    list(model.passage_embed(docs))

    if is_ci:
        delete_model_cache(model.model._model_dir)
