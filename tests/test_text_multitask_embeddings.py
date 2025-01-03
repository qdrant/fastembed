import os

import numpy as np
import pytest

from fastembed import TextEmbedding
from fastembed.text.multitask_embedding import Task
from tests.utils import delete_model_cache


CANONICAL_VECTOR_VALUES = {
    "jinaai/jina-embeddings-v3": [
        {
            "task_id": 0,
            "vectors": np.array(
                [
                    [0.0623, -0.0402, 0.1706, -0.0143, 0.0617],
                    [-0.1064, -0.0733, 0.0353, 0.0096, 0.0667],
                ]
            ),
        },
        {
            "task_id": 1,
            "vectors": np.array(
                [
                    [0.0513, -0.0247, 0.1751, -0.0075, 0.0679],
                    [-0.0987, -0.0786, 0.09, 0.0087, 0.0577],
                ]
            ),
        },
        {
            "task_id": 2,
            "vectors": np.array(
                [
                    [0.094, -0.1065, 0.1305, 0.0547, 0.0556],
                    [0.0315, -0.1468, 0.065, 0.0568, 0.0546],
                ]
            ),
        },
        {
            "task_id": 3,
            "vectors": np.array(
                [
                    [0.0606, -0.0877, 0.1384, 0.0065, 0.0722],
                    [-0.0502, -0.119, 0.032, 0.0514, 0.0689],
                ]
            ),
        },
        {
            "task_id": 4,
            "vectors": np.array(
                [
                    [0.0911, -0.0341, 0.1305, -0.026, 0.0576],
                    [-0.1432, -0.05, 0.0133, 0.0464, 0.0789],
                ]
            ),
        },
    ]
}
docs = ["Hello World", "Follow the white rabbit."]


def test_batch_embedding():
    is_ci = os.getenv("CI")
    docs_to_embed = docs * 10
    default_task = 4

    for model_desc in TextEmbedding.list_supported_models():
        if not is_ci and model_desc["size_in_GB"] > 1:
            continue

        model_name = model_desc["model"]
        dim = model_desc["dim"]

        if model_name not in CANONICAL_VECTOR_VALUES.keys():
            continue

        model = TextEmbedding(model_name=model_name)

        print(f"evaluating {model_name} default task")

        embeddings = list(model.embed(documents=docs_to_embed, batch_size=6))
        embeddings = np.stack(embeddings, axis=0)

        assert embeddings.shape == (len(docs_to_embed), dim)

        canonical_vector = CANONICAL_VECTOR_VALUES[model_name][default_task]["vectors"]
        assert np.allclose(
            embeddings[: len(docs), : canonical_vector.shape[1]], canonical_vector, atol=1e-4
        ), model_desc["model"]

        if is_ci:
            delete_model_cache(model.model._model_dir)


def test_single_embedding():
    is_ci = os.getenv("CI")

    for model_desc in TextEmbedding.list_supported_models():
        if not is_ci and model_desc["size_in_GB"] > 1:
            continue

        model_name = model_desc["model"]
        dim = model_desc["dim"]

        if model_name not in CANONICAL_VECTOR_VALUES.keys():
            continue

        model = TextEmbedding(model_name=model_name)

        for task in CANONICAL_VECTOR_VALUES[model_name]:
            print(f"evaluating {model_name} task_id: {task['task_id']}")

            embeddings = list(model.embed(documents=docs, task_id=task["task_id"]))
            embeddings = np.stack(embeddings, axis=0)

            assert embeddings.shape == (len(docs), dim)

            canonical_vector = task["vectors"]
            assert np.allclose(
                embeddings[: len(docs), : canonical_vector.shape[1]], canonical_vector, atol=1e-4
            ), model_desc["model"]

        if is_ci:
            delete_model_cache(model.model._model_dir)


def test_single_embedding_query():
    is_ci = os.getenv("CI")
    task_id = 0

    for model_desc in TextEmbedding.list_supported_models():
        if not is_ci and model_desc["size_in_GB"] > 1:
            continue

        model_name = model_desc["model"]
        dim = model_desc["dim"]

        if model_name not in CANONICAL_VECTOR_VALUES.keys():
            continue

        model = TextEmbedding(model_name=model_name)

        print(f"evaluating {model_name} query_embed task_id: {task_id}")

        embeddings = list(model.query_embed(query=docs))
        embeddings = np.stack(embeddings, axis=0)

        assert embeddings.shape == (len(docs), dim)

        canonical_vector = CANONICAL_VECTOR_VALUES[model_name][task_id]["vectors"]
        assert np.allclose(
            embeddings[: len(docs), : canonical_vector.shape[1]], canonical_vector, atol=1e-4
        ), model_desc["model"]

        if is_ci:
            delete_model_cache(model.model._model_dir)


def test_single_embedding_passage():
    is_ci = os.getenv("CI")
    task_id = 1

    for model_desc in TextEmbedding.list_supported_models():
        if not is_ci and model_desc["size_in_GB"] > 1:
            continue

        model_name = model_desc["model"]
        dim = model_desc["dim"]

        if model_name not in CANONICAL_VECTOR_VALUES.keys():
            continue

        model = TextEmbedding(model_name=model_name)

        print(f"evaluating {model_name} passage_embed task_id: {task_id}")

        embeddings = list(model.passage_embed(texts=docs))
        embeddings = np.stack(embeddings, axis=0)

        assert embeddings.shape == (len(docs), dim)

        canonical_vector = CANONICAL_VECTOR_VALUES[model_name][task_id]["vectors"]
        assert np.allclose(
            embeddings[: len(docs), : canonical_vector.shape[1]], canonical_vector, atol=1e-4
        ), model_desc["model"]

        if is_ci:
            delete_model_cache(model.model._model_dir)


def test_task_assignment():
    is_ci = os.getenv("CI")

    for model_desc in TextEmbedding.list_supported_models():
        if not is_ci and model_desc["size_in_GB"] > 1:
            continue

        model_name = model_desc["model"]
        if model_name not in CANONICAL_VECTOR_VALUES.keys():
            continue

        model = TextEmbedding(model_name=model_name)

        _ = list(model.embed(documents=docs, batch_size=1, task_id=2))
        assert model.model._current_task_id == Task.SEPARATION

        _ = list(
            model.embed(documents=docs, batch_size=1, parallel=1, task_id=Task.CLASSIFICATION)
        )
        assert model.model._current_task_id == 3

        _ = list(model.query_embed(query=docs))
        assert model.model._current_task_id == Task.RETRIEVAL_QUERY

        _ = list(model.passage_embed(texts=docs))
        assert model.model._current_task_id == Task.RETRIEVAL_PASSAGE

        if is_ci:
            delete_model_cache(model.model._model_dir)


@pytest.mark.parametrize(
    "model_name",
    ["jinaai/jina-embeddings-v3"],
)
def test_lazy_load(model_name):
    is_ci = os.getenv("CI")
    model = TextEmbedding(model_name=model_name, lazy_load=True)
    assert not hasattr(model.model, "model")

    list(model.embed(docs))
    assert hasattr(model.model, "model")

    if is_ci:
        delete_model_cache(model.model._model_dir)
