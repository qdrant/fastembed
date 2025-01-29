import os

import numpy as np
import pytest

from fastembed import SparseTextEmbedding
from tests.utils import delete_model_cache


@pytest.mark.parametrize("model_name", ["Qdrant/bm42-all-minilm-l6-v2-attentions", "Qdrant/bm25"])
def test_attention_embeddings(model_name) -> None:
    is_ci = os.getenv("CI")
    model = SparseTextEmbedding(model_name=model_name)

    output = list(
        model.query_embed(
            [
                "I must not fear. Fear is the mind-killer.",
            ]
        )
    )

    assert len(output) == 1

    for result in output:
        assert len(result.indices) == len(result.values)
        assert np.allclose(result.values, np.ones(len(result.values)))

    quotes = [
        "I must not fear. Fear is the mind-killer.",
        "All animals are equal, but some animals are more equal than others.",
        "It was a pleasure to burn.",
        "The sky above the port was the color of television, tuned to a dead channel.",
        "In the beginning, the universe was created."
        " This has made a lot of people very angry and been widely regarded as a bad move.",
        "It's a truth universally acknowledged that a zombie in possession of brains must be in want of more brains.",
        "War is peace. Freedom is slavery. Ignorance is strength.",
        "We're not in Infinity; we're in the suburbs.",
        "I was a thousand times more evil than thou!",
        "History is merely a list of surprises... It can only prepare us to be surprised yet again.",
        ".",  # Empty string
    ]

    output = list(model.embed(quotes))

    assert len(output) == len(quotes)

    for result in output[:-1]:
        assert len(result.indices) == len(result.values)
        assert len(result.indices) > 0

    assert len(output[-1].indices) == 0

    # Test support for unknown languages
    output = list(
        model.query_embed(
            [
                "привет мир!",
            ]
        )
    )

    assert len(output) == 1

    for result in output:
        assert len(result.indices) == len(result.values)
        assert len(result.indices) == 2

    if is_ci:
        delete_model_cache(model.model._model_dir)


@pytest.mark.parametrize("model_name", ["Qdrant/bm42-all-minilm-l6-v2-attentions", "Qdrant/bm25"])
def test_parallel_processing(model_name) -> None:
    is_ci = os.getenv("CI")

    model = SparseTextEmbedding(model_name=model_name)

    docs = ["hello world", "attention embedding", "Mangez-vous vraiment des grenouilles?"] * 100
    embeddings = list(model.embed(docs, batch_size=10, parallel=2))

    embeddings_2 = list(model.embed(docs, batch_size=10, parallel=None))

    embeddings_3 = list(model.embed(docs, batch_size=10, parallel=0))

    assert len(embeddings) == len(docs)

    for emb_1, emb_2, emb_3 in zip(embeddings, embeddings_2, embeddings_3):
        assert np.allclose(emb_1.indices, emb_2.indices)
        assert np.allclose(emb_1.indices, emb_3.indices)
        assert np.allclose(emb_1.values, emb_2.values)
        assert np.allclose(emb_1.values, emb_3.values)

    if is_ci:
        delete_model_cache(model.model._model_dir)


@pytest.mark.parametrize("model_name", ["Qdrant/bm25"])
def test_multilanguage(model_name) -> None:
    is_ci = os.getenv("CI")

    docs = ["Mangez-vous vraiment des grenouilles?", "Je suis au lit"]

    model = SparseTextEmbedding(model_name=model_name, language="french")
    embeddings = list(model.embed(docs))[:2]
    assert embeddings[0].values.shape == (3,)
    assert embeddings[0].indices.shape == (3,)

    assert embeddings[1].values.shape == (1,)
    assert embeddings[1].indices.shape == (1,)

    model = SparseTextEmbedding(model_name=model_name, language="english")
    embeddings = list(model.embed(docs))[:2]
    assert embeddings[0].values.shape == (5,)
    assert embeddings[0].indices.shape == (5,)

    assert embeddings[1].values.shape == (4,)
    assert embeddings[1].indices.shape == (4,)

    if is_ci:
        delete_model_cache(model.model._model_dir)


@pytest.mark.parametrize("model_name", ["Qdrant/bm25"])
def test_special_characters(model_name) -> None:
    is_ci = os.getenv("CI")

    docs = [
        "Über den größten Flüssen Österreichs äußern sich Experten häufig: Öko-Systeme müssen geschützt werden!",
        "L'élève français s'écrie : « Où est mon crayon ? J'ai besoin de finir cet exercice avant la récréation!",
        "Într-o zi însorită, Ștefan și Ioana au mâncat mămăligă cu brânză și au băut țuică la cabană.",
        "Üzgün öğretmen öğrencilere seslendi: Lütfen gürültü yapmayın, sınavınızı bitirmeye çalışıyorum!",
        "Ο Ξενοφών είπε: «Ψάχνω για ένα ωραίο δώρο για τη γιαγιά μου. Ίσως ένα φυτό ή ένα βιβλίο;»",
        "Hola! ¿Cómo estás? Estoy muy emocionado por el cumpleaños de mi hermano, ¡va a ser increíble! También quiero comprar un pastel de chocolate con fresas y un regalo especial: un libro titulado «Cien años de soledad",
    ]

    model = SparseTextEmbedding(model_name=model_name, language="english")
    embeddings = list(model.embed(docs))
    for idx, shape in enumerate([14, 18, 15, 10, 15]):
        assert embeddings[idx].values.shape == (shape,)
        assert embeddings[idx].indices.shape == (shape,)

    if is_ci:
        delete_model_cache(model.model._model_dir)


@pytest.mark.parametrize("model_name", ["Qdrant/bm42-all-minilm-l6-v2-attentions"])
def test_lazy_load(model_name) -> None:
    model = SparseTextEmbedding(model_name=model_name, lazy_load=True)
    assert not hasattr(model.model, "model")
    docs = ["hello world", "flag embedding"]
    list(model.embed(docs))
    assert hasattr(model.model, "model")

    model = SparseTextEmbedding(model_name=model_name, lazy_load=True)
    list(model.query_embed(docs))

    model = SparseTextEmbedding(model_name=model_name, lazy_load=True)
    list(model.passage_embed(docs))
