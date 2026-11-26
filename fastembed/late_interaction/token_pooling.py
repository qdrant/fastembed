from typing import cast

import numpy as np

from fastembed.common.types import NumpyArray

# Pooling needs at least this many unprotected tokens, since a single token has nothing to merge with
MINIMUM_NONPROTECTED_TOKENS = 2


def ward_groups(X: NumpyArray, n_clusters: int) -> list[list[int]]:
    """Group the rows of `X` with Ward linkage until `n_clusters` groups remain.

    Same partition as SciPy's `linkage(method="ward")` + `fcluster(criterion="maxclust")`.

    Args:
        X (NumpyArray): Observations, shape (n, m).
        n_clusters (int): Number of groups to stop at.

    Returns:
        list[list[int]]: Sorted row indices per group, groups ordered by their first index.

    Raises:
        ValueError: If `n_clusters` is smaller than 1.
    """
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be at least 1, got {n_clusters}")

    n = X.shape[0]
    gram = X @ X.T
    sq = gram.diagonal()
    # D[a, b] is twice the Ward cost of merging clusters a and b. Initially every cluster
    # is a single row, so that is just the squared euclidean distance between rows a and b.
    # The factor 2 is kept by the Lance-Williams update and does not change which pair is closest.
    # Rounding can push tiny entries below zero, hence the clip.
    D = np.maximum(sq[:, None] + sq[None, :] - 2 * gram, 0)
    # a cluster is never merged with itself
    np.fill_diagonal(D, np.inf)
    size = np.ones(n)
    members: list[list[int]] = [[i] for i in range(n)]

    for _ in range(n - n_clusters):
        # 1. Decide: the smallest entry of D is the cheapest pair to merge. `argmin` is flat
        # (no axis), so it returns an index into the n*n matrix, which we turn back into (row, col).
        i, j = np.unravel_index(np.argmin(D), D.shape)
        # D is symmetric, so every pair shows up as (i, j) and (j, i). Order them so that
        # the rule is always the same: keep the lower index i, drop j.
        if i > j:
            i, j = j, i

        # 2. Update costs: i now stands for the merged cluster i + j. `merge_cost[k]` is the
        # cost of merging it with every other cluster k, computed for all k at once with the
        # Lance-Williams update for Ward. `size` is the vector of cluster sizes.
        merge_cost = ((size[i] + size) * D[i] + (size[j] + size) * D[j] - size * D[i, j]) / (
            size[i] + size[j] + size
        )

        D[i, :] = merge_cost
        D[:, i] = merge_cost  # keep D symmetric
        # j has been merged into i, so it can never be picked again
        D[j, :] = np.inf
        D[:, j] = np.inf
        D[i, i] = np.inf

        # 3. Bookkeeping: i absorbs j
        size[i] += size[j]
        members[i] += members[j]
        members[j] = []

    groups = [m for m in members if len(m) > 0]
    # `members[i]` is built by concatenation, so sort each group
    return sorted((sorted(m) for m in groups), key=lambda m: m[0])


def pool_tokens_hierarchical(
    embeddings: NumpyArray, pool_factor: int, protected_tokens: int
) -> NumpyArray:
    """Pool the token vectors of one document: cluster them, then average each cluster.

    See https://www.answer.ai/posts/colbert-pooling.html. Pooled vectors are not re-normalized.

    Args:
        embeddings (NumpyArray): L2-normalized token embeddings, shape (n_tokens, dim).
        pool_factor (int): Keeps about `1 / pool_factor` of the vectors. <= 1 disables pooling.
        protected_tokens (int): Number of leading tokens (e.g. [CLS], [D]) to keep as they are.

    Returns:
        NumpyArray: At most `n_tokens // pool_factor + protected_tokens` vectors.
    """
    n = embeddings.shape[0]
    if pool_factor <= 1:
        return embeddings

    if n - protected_tokens < MINIMUM_NONPROTECTED_TOKENS:
        return embeddings

    protected = embeddings[:protected_tokens]
    rest = embeddings[protected_tokens:]

    n_clusters = max(1, n // pool_factor)
    if n_clusters >= rest.shape[0]:
        return embeddings

    # rows are unit length, so the dot product is the cosine similarity
    similarity = rest @ rest.T
    distance = 1 - similarity

    # the article runs Ward on the rows of this distance matrix, not on the embeddings
    groups = ward_groups(distance, n_clusters=n_clusters)
    pooled: list[NumpyArray] = [np.mean(rest[g], axis=0) for g in groups]
    return cast(NumpyArray, np.concatenate((protected, np.stack(pooled))))
