import copy

import pytest
from py_rust_stemmers import SnowballStemmer

from fastembed.sparse.utils.sparse_vectors_converter import SparseVectorConverter, WordEmbedding


@pytest.mark.parametrize("unknown_first", [False, True])
def test_clean_words_merges_counts_without_mutating_input(unknown_first: bool) -> None:
    converter = SparseVectorConverter(set(), SnowballStemmer("english"))
    known = WordEmbedding("cat", ["cat"], 1, 1, [1.0, 0.0])
    unknown = WordEmbedding("cat!", ["cat!"], 2, -1, [1.0])
    entries = [("cat", known), ("cat!", unknown)]
    sentence = dict(reversed(entries) if unknown_first else entries)
    original = copy.deepcopy(sentence)

    for _ in range(2):
        cleaned = converter.clean_words(sentence)

        assert cleaned["cat"].count == 3
        assert sorted(cleaned["cat"].forms) == ["cat", "cat!"]
        assert cleaned["cat"].word_id == known.word_id
        assert cleaned["cat"].embedding == known.embedding
        assert sentence == original
