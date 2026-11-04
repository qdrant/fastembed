import numpy as np
import pytest

from fastembed import (
    TextEmbedding,
    SparseTextEmbedding,
    ImageEmbedding,
    LateInteractionMultimodalEmbedding,
    LateInteractionTextEmbedding,
)
from fastembed.common.utils import (
    iter_batch,
    last_token_pooling,
    mean_pooling,
    normalize,
    remove_non_alphanumeric,
)
from fastembed.sparse.utils.tokenizer import SimpleTokenizer


def test_text_list_supported_models():
    for model_type in [
        TextEmbedding,
        SparseTextEmbedding,
        ImageEmbedding,
        LateInteractionMultimodalEmbedding,
        LateInteractionTextEmbedding,
    ]:
        supported_models = model_type.list_supported_models()
        assert isinstance(supported_models, list)
        description = supported_models[0]
        assert isinstance(description, dict)

        assert "model" in description and description["model"]
        if model_type != SparseTextEmbedding:
            assert "dim" in description and description["dim"]
        assert "license" in description and description["license"]
        assert "size_in_GB" in description and description["size_in_GB"]
        assert "model_file" in description and description["model_file"]
        assert "sources" in description and description["sources"]
        assert "hf" in description["sources"] or "url" in description["sources"]


def test_last_token_pooling():
    token_embeddings = np.array(
        [
            [[1.0, 1.0], [2.0, 2.0], [9.0, 9.0], [9.0, 9.0]],  # 2 real tokens, then padding
            [[3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],  # no padding
        ]
    )
    attention_mask = np.array([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=np.int64)

    pooled = last_token_pooling(token_embeddings, attention_mask)

    assert np.allclose(pooled, [[2.0, 2.0], [6.0, 6.0]])


def test_last_token_pooling_with_left_padding():
    token_embeddings = np.array(
        [
            [[9.0, 9.0], [9.0, 9.0], [1.0, 1.0], [2.0, 2.0]],  # padding, then 2 real tokens
            [[3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],  # no padding
        ]
    )
    attention_mask = np.array([[0, 0, 1, 1], [1, 1, 1, 1]], dtype=np.int64)

    pooled = last_token_pooling(token_embeddings, attention_mask)

    assert np.allclose(pooled, [[2.0, 2.0], [6.0, 6.0]])


def test_mean_pooling():
    # 2 real tokens, then padding
    token_embeddings = np.array([[[1.0, 2.0], [3.0, 4.0], [9.0, 9.0]]], dtype=np.float32)
    attention_mask = np.array([[1, 1, 0]], dtype=np.int64)
    # the sum over 8192 tokens of 10.0 exceeds the float16 max of 65504
    long_sequence = np.full((1, 8192, 2), 10.0, dtype=np.float16)

    pooled = mean_pooling(token_embeddings, attention_mask)
    pooled_long_sequence = mean_pooling(long_sequence, np.ones((1, 8192), dtype=np.int64))

    assert pooled.dtype == pooled_long_sequence.dtype == np.float64
    assert np.array_equal(pooled, [[2.0, 3.0]])
    assert np.array_equal(pooled_long_sequence, [[10.0, 10.0]])


def test_normalize_does_not_overflow_float16():
    # the sum of squares, 1024 * 10.0**2, exceeds the float16 max of 65504
    embeddings = np.full((1, 1024), 10.0, dtype=np.float16)

    normalized = normalize(embeddings)

    assert normalized.dtype == np.float16
    assert np.array_equal(normalized, np.full((1, 1024), 1 / 32))


def test_iter_batch_accepts_positive_size():
    assert list(iter_batch([1, 2, 3, 4, 5], 3)) == [[1, 2, 3], [4, 5]]


def test_iter_batch_rejects_non_positive_size():
    with pytest.raises(ValueError, match="batch_size must be >= 1, got 0"):
        iter_batch([1, 2, 3], 0)

    with pytest.raises(ValueError, match="batch_size must be >= 1, got -1"):
        iter_batch([1, 2, 3], -1)


def test_remove_non_alphanumeric_legacy_default_strips_combining_marks():
    # Existing BM25 indexes hashed cafe\u0301 as cafe. The default must keep that.
    assert remove_non_alphanumeric("cafe\u0301") == "cafe "
    assert remove_non_alphanumeric("café") == "café"


def test_remove_non_alphanumeric_opt_in_keeps_combining_marks_inside_words():
    # Tamil and Devanagari vowel signs and Arabic harakat are combining marks (Unicode
    # category M). The word class does not match them, so a pattern that keeps only word
    # characters cuts every word in these scripts at each mark.
    assert remove_non_alphanumeric("தமிழ், हिन्दी!", preserve_combining_marks=True) == "தமிழ்  हिन्दी "
    assert remove_non_alphanumeric("cafe\u0301", preserve_combining_marks=True) == "café"


def test_tokenize_legacy_default_strips_combining_marks():
    assert SimpleTokenizer.tokenize("cafe\u0301") == ["cafe"]
    assert SimpleTokenizer.tokenize("café") == ["café"]


def test_tokenize_opt_in_keeps_combining_marks_inside_words():
    text = "தமிழ் மொழி, हिन्दी भाषा! ذَهَبَ الطَّالِبُ"

    assert SimpleTokenizer.tokenize(text, preserve_combining_marks=True) == [
        "தமிழ்",
        "மொழி",
        "हिन्दी",
        "भाषा",
        "ذَهَبَ",
        "الطَّالِبُ",
    ]
    assert SimpleTokenizer.tokenize("cafe\u0301", preserve_combining_marks=True) == ["café"]


def test_tokenize_opt_in_nfc_unifies_equivalent_forms():
    decomposed = SimpleTokenizer.tokenize("cafe\u0301", preserve_combining_marks=True)
    precomposed = SimpleTokenizer.tokenize("café", preserve_combining_marks=True)
    acute_then_dot = SimpleTokenizer.tokenize("a\u0301\u0323", preserve_combining_marks=True)
    dot_then_acute = SimpleTokenizer.tokenize("a\u0323\u0301", preserve_combining_marks=True)

    assert decomposed == precomposed == ["café"]
    assert acute_then_dot == dot_then_acute
    assert acute_then_dot != ["a"]


def test_tokenize_opt_in_strips_orphan_combining_marks():
    # U+2764 U+FE0F is the heart emoji; the variation selector is a mark with no
    # letter/digit base and must not attach to the following word.
    assert SimpleTokenizer.tokenize("\u2764\ufe0fhello", preserve_combining_marks=True) == [
        "hello"
    ]
    assert SimpleTokenizer.tokenize("\ufe0fhello", preserve_combining_marks=True) == ["hello"]
    assert SimpleTokenizer.tokenize("\ufe0f", preserve_combining_marks=True) == []


def test_tokenize_opt_in_separates_keycap_emoji_from_following_word():
    # Digit + VS16 + enclosing keycap is the 1️⃣ emoji and must not glue to hello.
    assert SimpleTokenizer.tokenize("1\ufe0f\u20e3hello", preserve_combining_marks=True) == [
        "1",
        "hello",
    ]


def test_tokenize_opt_in_maps_dotted_capital_i_to_ascii_i():
    assert SimpleTokenizer.tokenize("İstanbul", preserve_combining_marks=True) == ["istanbul"]
    assert SimpleTokenizer.tokenize("istanbul", preserve_combining_marks=True) == ["istanbul"]


def test_tokenize_opt_in_keeps_lowercase_lithuanian_dotted_i():
    # A combining dot already present in lowercase text is not an U+0130 artifact.
    assert SimpleTokenizer.tokenize("i\u0307", preserve_combining_marks=True) == ["i\u0307"]
    assert SimpleTokenizer.tokenize("i\u0307\u0301", preserve_combining_marks=True) != ["i\u0301"]


def test_tokenize_legacy_dotted_i_still_splits_on_combining_dot():
    assert SimpleTokenizer.tokenize("İstanbul") == ["i", "stanbul"]


def test_remove_non_alphanumeric_opt_in_strips_orphan_combining_marks():
    assert remove_non_alphanumeric("\u2764\ufe0fhello", preserve_combining_marks=True) == "  hello"


def test_remove_non_alphanumeric_leaves_ascii_alphanumeric_queries_intact():
    # The fast path for ASCII text must keep the previous word/whitespace semantics.
    assert remove_non_alphanumeric("Errore 0x000006D9") == "Errore 0x000006D9"
    assert remove_non_alphanumeric("hello, world!") == "hello  world "
