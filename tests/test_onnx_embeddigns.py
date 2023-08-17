from fastembed.embedding import DefaultEmbedding


def test_onnx_inference():
    model = DefaultEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    docs = [
        "hello world"
    ]

    expected_vector = [
        -3.44772711e-02,  3.10231801e-02,  6.73499797e-03,  2.61089597e-02
    ]

    vector = list(model.encode(docs))[0]
    assert len(vector) == 384

    for i in range(len(expected_vector)):
        assert abs(vector[i] - expected_vector[i]) < 1e-3




