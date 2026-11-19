import numpy as np
import pytest
from py_rust_stemmers import SnowballStemmer

from fastembed.sparse.utils.sparse_vectors_converter import SparseVectorConverter, WordEmbedding


@pytest.mark.parametrize("query", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_minicoil_merges_colliding_oov_indices(query: bool, reverse: bool) -> None:
    converter = SparseVectorConverter(set(), SnowballStemmer("english"), k=1.0, b=0.0)
    # Distinct alphanumeric words with a real MurmurHash collision, not a mocked hash.
    words = [
        WordEmbedding("token49516", ["token49516"], 1, -1, [1.0]),
        WordEmbedding("token56101", ["token56101"], 2, -1, [1.0]),
    ]
    if reverse:
        words.reverse()
    sentence = {word.word: word for word in words}
    convert = converter.embedding_to_vector_query if query else converter.embedding_to_vector

    vector = convert(sentence, embedding_size=4, vocab_size=10000)

    assert vector.indices.tolist() == [1433340619]
    assert vector.values.tolist() == pytest.approx([2.0 if query else 7 / 3])
    assert vector.indices.dtype == np.int32
    assert vector.values.dtype == np.float32


@pytest.mark.parametrize("query", [False, True])
def test_minicoil_preserves_noncolliding_coordinates(query: bool) -> None:
    converter = SparseVectorConverter(set(), SnowballStemmer("english"), k=1.0, b=0.0)
    sentence = {
        "cat": WordEmbedding("cat", ["cat"], 2, 7, [3.0, 4.0]),
        "token49516": WordEmbedding("token49516", ["token49516"], 1, -1, [1.0]),
    }
    convert = converter.embedding_to_vector_query if query else converter.embedding_to_vector

    vector = convert(sentence, embedding_size=2, vocab_size=10000)

    assert vector.indices.tolist() == [14, 15, 1433308619]
    assert vector.values.tolist() == pytest.approx(
        [0.6, 0.8, 1.0] if query else [0.8, 16 / 15, 1.0]
    )


@pytest.mark.parametrize("query", [False, True])
@pytest.mark.parametrize("filtered", [False, True])
def test_minicoil_preserves_empty_vectors(query: bool, filtered: bool) -> None:
    converter = SparseVectorConverter({"the"}, SnowballStemmer("english"))
    sentence = {"the": WordEmbedding("the", ["the"], 1, -1, [1.0])} if filtered else {}
    convert = converter.embedding_to_vector_query if query else converter.embedding_to_vector

    vector = convert(sentence, embedding_size=4, vocab_size=10000)

    assert vector.indices.size == vector.values.size == 0
    assert vector.indices.dtype == np.int32
    assert vector.values.dtype == np.float32
