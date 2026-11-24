import re
from pathlib import Path

import pytest

from fastembed import SparseTextEmbedding
from fastembed.sparse.bm25 import Bm25


class _ChineseTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"北京|大学|搜索|\S", text)


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    (tmp_path / "english.txt").write_text("the\nand\n", encoding="utf-8")
    return tmp_path


def test_custom_tokenizer_applies_to_documents_queries_and_token_count(model_dir: Path) -> None:
    model = SparseTextEmbedding(
        "Qdrant/bm25",
        specific_model_path=str(model_dir),
        language="chinese",
        disable_stemmer=True,
        tokenizer=_ChineseTokenizer(),
        stopwords={"搜索"},
        k=1.0,
        b=0.0,
    )
    text = "北京大学北京搜索！"
    document = next(iter(model.embed(text)))
    query = next(iter(model.query_embed(text)))

    assert document.as_dict() == pytest.approx(
        {Bm25.compute_token_id("北京"): 4 / 3, Bm25.compute_token_id("大学"): 1.0}
    )
    assert query.as_dict() == {
        Bm25.compute_token_id("北京"): 1.0,
        Bm25.compute_token_id("大学"): 1.0,
    }
    assert model.token_count(text) == 4
    assert model.token_count(iter([text, "北京大学"])) == 6


@pytest.mark.parametrize("parallel", [None, 2])
def test_custom_tokenizer_is_preserved_in_worker_processes(
    model_dir: Path, parallel: int | None
) -> None:
    model = SparseTextEmbedding(
        "Qdrant/bm25",
        specific_model_path=str(model_dir),
        language="chinese",
        disable_stemmer=True,
        tokenizer=_ChineseTokenizer(),
    )
    documents = ["北京大学", "搜索北京", "", "大学大学"]
    expected = list(model.embed(documents))
    actual = list(model.embed(iter(documents), batch_size=1, parallel=parallel))

    assert len(actual) == len(expected) == len(documents)
    for serial, result in zip(expected, actual):
        assert result.indices.tolist() == serial.indices.tolist()
        assert result.values.tolist() == pytest.approx(serial.values.tolist())
    assert set(actual[0].indices.tolist()) == {
        Bm25.compute_token_id("北京"),
        Bm25.compute_token_id("大学"),
    }


@pytest.mark.parametrize("tokenizer", [None, _ChineseTokenizer()])
def test_empty_input_with_optional_tokenizer(
    model_dir: Path, tokenizer: _ChineseTokenizer | None
) -> None:
    model = Bm25(
        "Qdrant/bm25",
        specific_model_path=str(model_dir),
        disable_stemmer=True,
        tokenizer=tokenizer,
    )

    assert list(model.embed([])) == []
    assert next(iter(model.embed(""))).as_dict() == {}
    assert next(iter(model.query_embed(""))).as_dict() == {}
    assert model.token_count("") == 0


def test_default_tokenizer_preserves_english_stemming_and_stopwords(model_dir: Path) -> None:
    model = Bm25("Qdrant/bm25", specific_model_path=str(model_dir), k=1.0, b=0.0)

    assert next(iter(model.embed("The foxes and cats!"))).as_dict() == {
        Bm25.compute_token_id("fox"): 1.0,
        Bm25.compute_token_id("cat"): 1.0,
    }
    assert model.token_count("The foxes and cats!") == 4
