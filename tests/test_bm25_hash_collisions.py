"""Regression tests for order-independent BM25 weights when token hashes collide."""

from collections import Counter
from pathlib import Path

import mmh3
import pytest

from fastembed.sparse.bm25 import Bm25


COLLIDING_TOKENS = ("kitchens", "prostaglandins")
COLLIDING_TOKEN_ID = 358434922


@pytest.fixture
def model() -> Bm25:
    """Create only the BM25 state required to calculate term-frequency weights."""
    instance = Bm25.__new__(Bm25)
    instance.k = 1.2
    instance.b = 0.75
    instance.avg_len = 256.0
    return instance


def _assert_bag_weights_are_order_independent(model: Bm25, tokens: list[str]) -> None:
    """Check that permutations preserve hash IDs and weights within rounding tolerance."""
    expected_keys = {model.compute_token_id(token) for token in tokens}
    reference = model._term_frequency(tokens)
    assert set(reference) == expected_keys
    for reordered in (
        list(reversed(tokens)),
        sorted(tokens),
        sorted(tokens, reverse=True),
        tokens[1:] + tokens[:1],
    ):
        result = model._term_frequency(reordered)
        assert set(result) == expected_keys
        assert result == pytest.approx(reference, rel=1e-12, abs=1e-12)


def test_real_tokens_collide_after_absolute_value_of_signed_hash() -> None:
    """Verify a real token pair with opposite signed hashes and the same sparse ID."""
    assert mmh3.hash("kitchens") == COLLIDING_TOKEN_ID
    assert mmh3.hash("prostaglandins") == -COLLIDING_TOKEN_ID
    assert {Bm25.compute_token_id(token) for token in COLLIDING_TOKENS} == {COLLIDING_TOKEN_ID}


@pytest.mark.parametrize("counts", [(1, 2), (3, 1), (2, 5), (7, 3)])
def test_real_collision_weights_are_independent_of_token_order(
    model: Bm25, counts: tuple[int, int]
) -> None:
    """Keep unequal colliding token counts independent of first-appearance order."""
    tokens = [COLLIDING_TOKENS[0]] * counts[0] + [COLLIDING_TOKENS[1]] * counts[1]

    _assert_bag_weights_are_order_independent(model, tokens)


def test_real_collision_with_other_tokens_preserves_keys_and_order_independence(
    model: Bm25,
) -> None:
    """Preserve sparse IDs and order independence when collisions mix with other tokens."""
    tokens = ["kitchens", "hello", "kitchens", "prostaglandins", "hello", "kitchens"]

    _assert_bag_weights_are_order_independent(model, tokens)


@pytest.mark.parametrize("counts", [(1, 2, 3), (2, 4, 1), (5, 1, 2)])
def test_three_distinct_colliding_tokens_are_order_independent(
    model: Bm25, monkeypatch: pytest.MonkeyPatch, counts: tuple[int, int, int]
) -> None:
    """Cover controlled three-way collisions without fixing a collision scoring policy."""
    colliding = ("alpha", "beta", "gamma")
    original_hash = model.compute_token_id

    def controlled_hash(token: str) -> int:
        """Force selected tokens to collide while preserving ordinary token hashes."""
        return COLLIDING_TOKEN_ID if token in colliding else original_hash(token)

    monkeypatch.setattr(model, "compute_token_id", controlled_hash)
    tokens = [token for token, count in zip(colliding, counts) for _ in range(count)]
    tokens += ["hello", "hello"]

    _assert_bag_weights_are_order_independent(model, tokens)


@pytest.mark.parametrize(
    "tokens",
    [
        ["hello"],
        ["hello", "hello", "hello"],
        ["hello", "world"],
        ["hello", "world", "hello", "third", "world", "world"],
    ],
)
def test_noncolliding_tokens_preserve_bm25_term_frequency_formula(
    model: Bm25, tokens: list[str]
) -> None:
    """Preserve the standard BM25 formula for ordinary and repeated noncolliding tokens."""
    counts = Counter(tokens)
    token_ids = {token: model.compute_token_id(token) for token in counts}
    assert len(set(token_ids.values())) == len(counts)
    expected = {
        token_ids[token]: count
        * (model.k + 1)
        / (count + model.k * (1 - model.b + model.b * len(tokens) / model.avg_len))
        for token, count in counts.items()
    }

    assert model._term_frequency(tokens) == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_empty_tokens_have_no_term_frequencies(model: Bm25) -> None:
    """Return an empty weight mapping when there are no document tokens."""
    assert model._term_frequency([]) == {}


def test_document_embeddings_of_the_same_colliding_token_bag_are_order_independent(
    tmp_path: Path,
) -> None:
    """Keep actual document embeddings stable when colliding tokens are reordered."""
    model = Bm25(
        "Qdrant/bm25",
        cache_dir=str(tmp_path),
        specific_model_path=str(tmp_path),
        disable_stemmer=True,
        local_files_only=True,
    )
    embeddings = list(
        model.embed(["kitchens kitchens prostaglandins", "prostaglandins kitchens kitchens"])
    )

    assert len(embeddings) == 2
    assert set(embeddings[0].as_dict()) == set(embeddings[1].as_dict()) == {COLLIDING_TOKEN_ID}
    assert embeddings[0].as_dict() == pytest.approx(embeddings[1].as_dict(), rel=1e-12, abs=1e-12)
