from fastembed.sparse.bm42 import Bm42


def test_attention_embeddings():
    model = Bm42(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")

    output = list(model.query_embed([
        "I must not fear. Fear is the mind-killer.",
    ]))

    assert len(output) == 1

    for result in output:
        assert len(result.indices) == len(result.values)

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
    ]

    output = list(model.embed(quotes))

    assert len(output) == len(quotes)

    for result in output:
        assert len(result.indices) == len(result.values)
        assert len(result.indices) > 0

    # Test support for unknown languages
    output = list(model.query_embed([
        "привет мир!",
    ]))

    assert len(output) == 1

    for result in output:
        assert len(result.indices) == len(result.values)
        assert len(result.indices) == 2
