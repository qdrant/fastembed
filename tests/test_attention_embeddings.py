from fastembed.sparse.bm42 import Bm42


def test_attention_embeddings():
    model = Bm42(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")

    output = list(model.embed(["With great power comes great responsebility"]))[0]

    print(output)
