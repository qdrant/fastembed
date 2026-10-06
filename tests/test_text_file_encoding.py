"""Model files are UTF-8; reading them with the platform default (cp1252 on Windows) breaks non-ASCII text."""

import json
from pathlib import Path

import pytest
from py_rust_stemmers import SnowballStemmer

from fastembed.common.preprocessor_utils import load_special_tokens
from fastembed.sparse.bm25 import Bm25
from fastembed.sparse.utils.vocab_resolver import VocabResolver, VocabTokenizerBase

# Stopwords from the files Qdrant/bm25 ships for these languages. Under cp1252 the German and
# French ones load as mojibake ("Ã¼ber") and the others raise UnicodeDecodeError.
NON_ASCII_STOPWORDS = {
    "german": ["aber", "über", "würde"],
    "french": ["à", "été", "au"],
    "russian": ["и", "в", "во"],
    "greek": ["αλλα", "αν", "αντι"],
    "arabic": ["إذ", "إذا", "إذما"],
}


@pytest.mark.parametrize(("language", "stopwords"), NON_ASCII_STOPWORDS.items())
def test_bm25_loads_non_ascii_stopwords(
    tmp_path: Path, language: str, stopwords: list[str]
) -> None:
    (tmp_path / f"{language}.txt").write_text("\n".join(stopwords), encoding="utf-8")

    assert Bm25._load_stopwords(tmp_path, language) == stopwords


def test_vocab_resolver_round_trips_non_ascii_words(tmp_path: Path) -> None:
    words = ["café", "naïve", "über", "straße"]
    resolver = VocabResolver(VocabTokenizerBase(), set(), SnowballStemmer("english"))
    for word in words:
        resolver.add_word(word)

    resolver.save_vocab(str(tmp_path / "vocab.txt"))
    resolver.save_json_vocab(str(tmp_path / "vocab.json"))
    assert (tmp_path / "vocab.txt").read_text(encoding="utf-8").splitlines() == words

    from_txt = VocabResolver(VocabTokenizerBase(), set(), SnowballStemmer("english"))
    from_txt.load_vocab(str(tmp_path / "vocab.txt"))
    from_json = VocabResolver(VocabTokenizerBase(), set(), SnowballStemmer("english"))
    from_json.load_json_vocab(str(tmp_path / "vocab.json"))

    for loaded in (from_txt, from_json):
        assert loaded.words == words
        assert loaded.vocab == resolver.vocab
        assert loaded.stem_mapping == resolver.stem_mapping


def test_vocab_resolver_reads_unescaped_utf8_json(tmp_path: Path) -> None:
    # json.dump escapes non-ASCII by default, but a vocab file written elsewhere may not
    path = tmp_path / "vocab.json"
    path.write_text(
        json.dumps({"vocab": ["café"], "stem_mapping": {"café": "café"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    resolver = VocabResolver(VocabTokenizerBase(), set(), SnowballStemmer("english"))

    resolver.load_json_vocab(str(path))

    assert resolver.vocab == {"café": 1}


def test_load_special_tokens_reads_utf8(tmp_path: Path) -> None:
    # SentencePiece-based tokenizers use "▁" (U+2581), which cp1252 cannot decode
    tokens_map = {"unk_token": "<unk>", "additional_special_tokens": ["▁<extra>"]}
    (tmp_path / "special_tokens_map.json").write_text(
        json.dumps(tokens_map, ensure_ascii=False), encoding="utf-8"
    )

    assert load_special_tokens(tmp_path) == tokens_map
