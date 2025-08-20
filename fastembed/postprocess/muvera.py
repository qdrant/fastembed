from typing import Union

import numpy as np

from fastembed.common.types import NumpyArray
from fastembed.late_interaction.late_interaction_embedding_base import (
    LateInteractionTextEmbeddingBase,
)
from fastembed.late_interaction_multimodal.late_interaction_multimodal_embedding_base import (
    LateInteractionMultimodalEmbeddingBase,
)


MultiVectorModel = Union[LateInteractionTextEmbeddingBase, LateInteractionMultimodalEmbeddingBase]
MAX_HAMMING_DISTANCE = 65  # 64 bits + 1
POPCOUNT_LUT = np.array([bin(x).count("1") for x in range(256)], dtype=np.uint8)


def hamming_distance_matrix(ids: np.ndarray) -> np.ndarray:
    """Compute full Hamming distance matrix

    Args:
    ids: shape (n,) - array of ids, only size of the array matters

    Return:
        np.ndarray (n, n) - hamming distance matrix
    """
    n = len(ids)
    xor_vals = np.bitwise_xor(ids[:, None], ids[None, :])  # (n, n) uint64
    bytes_view = xor_vals.view(np.uint8).reshape(n, n, 8)  # (n, n, 8)
    return POPCOUNT_LUT[bytes_view].sum(axis=2)


class SimHashProjection:
    """
    SimHash projection component for MUVERA clustering.

    This class implements locality-sensitive hashing using random hyperplanes
    to partition the vector space into 2^k_sim clusters. Each vector is assigned
    to a cluster based on which side of k_sim random hyperplanes it falls on.

    Attributes:
        k_sim (int): Number of SimHash functions (hyperplanes)
        dim (int): Dimensionality of input vectors
        simhash_vectors (np.ndarray): Random hyperplane normal vectors of shape (dim, k_sim)
    """

    def __init__(self, k_sim: int, dim: int, random_generator: np.random.Generator):
        """
        Initialize SimHash projection with random hyperplanes.

        Args:
            k_sim (int): Number of SimHash functions, determines 2^k_sim clusters
            dim (int): Dimensionality of input vectors
            random_generator (np.random.Generator): Random number generator for reproducibility
        """
        self.k_sim = k_sim
        self.dim = dim
        # Generate k_sim random hyperplanes (normal vectors) from standard normal distribution
        self.simhash_vectors = random_generator.normal(size=(dim, k_sim))

    def get_cluster_ids(self, vectors: np.ndarray) -> np.ndarray:
        """
        Compute the cluster IDs for a given vector using SimHash.

        The cluster ID is determined by computing the dot product of the vector
        with each hyperplane normal vector, taking the sign, and interpreting
        the resulting binary string as an integer.

        Args:
            vectors (np.ndarray): Input vectors of shape (n, dim,)

        Returns:
            np.ndarray: Cluster IDs in range [0, 2^k_sim - 1]

        Raises:
            AssertionError: If a vector shape doesn't match expected dimensionality
        """
        dot_product = (
            vectors @ self.simhash_vectors
        )  # (token_num, dim) x (dim, k_sim) -> (token_num, k_sim)
        cluster_ids = (dot_product > 0) @ (1 << np.arange(self.k_sim))
        return cluster_ids


class Muvera:
    """
    MUVERA (Multi-Vector Retrieval Architecture) algorithm implementation.

    This class creates Fixed Dimensional Encodings (FDEs) from variable-length
    sequences of vectors by using SimHash clustering and random projections.
    The process involves:
    1. Clustering vectors using multiple SimHash projections
    2. Computing cluster centers (with different strategies for docs vs queries)
    3. Applying random projections for dimensionality reduction
    4. Concatenating results from all projections

    Attributes:
        k_sim (int): Number of SimHash functions per projection
        dim (int): Input vector dimensionality
        dim_proj (int): Output dimensionality after random projection
        r_reps (int): Number of random projection repetitions
        random_seed (int): Random seed for consistent random matrix generation
        simhash_projections (List[SimHashProjection]): SimHash instances for clustering
        dim_reduction_projections (np.ndarray): Random projection matrices of shape (R_reps, d, d_proj)
    """

    def __init__(
        self,
        dim: int,
        k_sim: int = 5,
        dim_proj: int = 16,
        r_reps: int = 20,
        random_seed: int = 42,
    ):
        """
        Initialize MUVERA algorithm with specified parameters.

        Args:
            dim (int): Dimensionality of individual input vectors
            k_sim (int, optional): Number of SimHash functions (creates 2^k_sim clusters).
                                   Defaults to 5.
            dim_proj (int, optional): Dimensionality after random projection (must be <= dim).
                                    Defaults to 16.
            r_reps (int, optional): Number of random projection repetitions for robustness.
                                    Defaults to 20.
            random_seed (int, optional): Seed for random number generator to ensure
                                         reproducible results. Defaults to 42.

        Raises:
            ValueError: If dim_proj > dim (cannot project to higher dimensionality)
        """
        if dim_proj > dim:
            raise ValueError(
                f"Cannot project to a higher dimensionality (dim_proj={dim_proj} > dim={dim})"
            )

        self.k_sim = k_sim
        self.dim = dim
        self.dim_proj = dim_proj
        self.r_reps = r_reps
        # Create r_reps independent SimHash projections for robustness
        generator = np.random.default_rng(random_seed)
        self.simhash_projections = [
            SimHashProjection(k_sim=self.k_sim, dim=self.dim, random_generator=generator)
            for _ in range(r_reps)
        ]
        # Random projection matrices with entries from {-1, +1} for each repetition
        self.dim_reduction_projections = generator.choice([-1, 1], size=(r_reps, dim, dim_proj))

    @classmethod
    def from_multivector_model(
        cls,
        model: MultiVectorModel,
        k_sim: int = 5,
        dim_proj: int = 16,
        r_reps: int = 20,  # noqa[naming]
        random_seed: int = 42,
    ) -> "Muvera":
        """
        Create a Muvera instance from a multi-vector embedding model.

        This class method provides a convenient way to initialize a MUVERA
        that is compatible with a given multi-vector model by automatically extracting
        the embedding dimensionality from the model.

        Args:
            model (MultiVectorModel): A late interaction text or multimodal embedding model
                                    that provides multi-vector embeddings. Must have an
                                    `embedding_size` attribute specifying the dimensionality
                                    of individual vectors.
            k_sim (int, optional): Number of SimHash functions (creates 2^k_sim clusters).
                                   Defaults to 5.
            dim_proj (int, optional): Dimensionality after random projection (must be <= model's
                                    embedding_size). Defaults to 16.
            r_reps (int, optional): Number of random projection repetitions for robustness.
                                    Defaults to 20.
            random_seed (int, optional): Seed for random number generator to ensure
                                         reproducible results. Defaults to 42.

        Returns:
            Muvera: A configured MUVERA instance ready to process embeddings from the given model.

        Raises:
            ValueError: If dim_proj > model.embedding_size (cannot project to higher dimensionality)

        Example:
            >>> from fastembed import LateInteractionTextEmbedding
            >>> model = LateInteractionTextEmbedding(model_name="colbert-ir/colbertv2.0")
            >>> muvera = Muvera.from_multivector_model(
            ...     model=model,
            ...     k_sim=6,
            ...     dim_proj=32
            ... )
            >>> # Now use postprocessor with embeddings from the model
            >>> embeddings = np.array(list(model.embed(["sample text"])))
            >>> fde = muvera.process_document(embeddings[0])
        """
        return cls(
            dim=model.embedding_size,
            k_sim=k_sim,
            dim_proj=dim_proj,
            r_reps=r_reps,
            random_seed=random_seed,
        )

    def _get_output_dimension(self) -> int:
        """
        Get the output dimension of the MUVERA algorithm.

        Returns:
            int: Output dimension (r_reps * num_partitions * dim_proj) where b = 2^k_sim
        """
        num_partitions = 2**self.k_sim
        return self.r_reps * num_partitions * self.dim_proj

    @property
    def embedding_size(self) -> int:
        return self._get_output_dimension()

    def process_document(self, vectors: NumpyArray) -> NumpyArray:
        """
        Encode a document's vectors into a Fixed Dimensional Encoding (FDE).

        Uses document-specific settings: normalizes cluster centers by vector count
        and fills empty clusters using Hamming distance-based selection.

        Args:
            vectors (NumpyArray): Document vectors of shape (n_tokens, dim)

        Returns:
            NumpyArray: Fixed dimensional encodings of shape (r_reps * b * dim_proj,)
        """
        return self.process(vectors, fill_empty_clusters=True, normalize_by_count=True)

    def process_query(self, vectors: NumpyArray) -> NumpyArray:
        """
        Encode a query's vectors into a Fixed Dimensional Encoding (FDE).

        Uses query-specific settings: no normalization by count and no empty
        cluster filling to preserve query vector magnitudes.

        Args:
            vectors (NumpyArray]): Query vectors of shape (n_tokens, dim)

        Returns:
            NumpyArray: Fixed dimensional encoding of shape (r_reps * b * dim_proj,)
        """
        return self.process(vectors, fill_empty_clusters=False, normalize_by_count=False)

    def process(
        self,
        vectors: NumpyArray,
        fill_empty_clusters: bool = True,
        normalize_by_count: bool = True,
    ) -> NumpyArray:
        """
        Core encoding method that transforms variable-length vector sequences into FDEs.

        The encoding process:
        1. For each of r_reps random projections:
           a. Assign vectors to clusters using SimHash
           b. Compute cluster centers (sum of vectors in each cluster)
           c. Optionally normalize by cluster size
           d. Fill empty clusters using Hamming distance if requested
           e. Apply random projection for dimensionality reduction
           f. Flatten cluster centers into a vector
        2. Concatenate all projection results

        Args:
            vectors (np.ndarray): Input vectors of shape (n_vectors, dim)
            fill_empty_clusters (bool): Whether to fill empty clusters using nearest
                                      vectors based on Hamming distance of cluster IDs
            normalize_by_count (bool): Whether to normalize cluster centers by the
                                     number of vectors assigned to each cluster

        Returns:
            np.ndarray: Fixed dimensional encoding of shape (r_reps * b * dim_proj)
                        where B = 2^k_sim is the number of clusters

        Raises:
            AssertionError: If input vectors don't have expected dimensionality
        """
        assert (
            vectors.shape[1] == self.dim
        ), f"Expected vectors of shape (n, {self.dim}), got {vectors.shape}"

        # Store results from each random projection
        output_vectors = []

        # num of space partitions in SimHash
        num_partitions = 2**self.k_sim
        cluster_center_ids = np.arange(num_partitions)
        precomputed_hamming_matrix = (
            hamming_distance_matrix(cluster_center_ids) if fill_empty_clusters else None
        )

        for projection_index, simhash in enumerate(self.simhash_projections):
            # Initialize cluster centers and count vectors assigned to each cluster
            cluster_centers = np.zeros((num_partitions, self.dim))
            cluster_center_id_to_vectors: dict[int, list[int]] = {
                cluster_center_id: [] for cluster_center_id in cluster_center_ids
            }
            cluster_vector_counts = None
            empty_mask = None

            # Assign each vector to its cluster and accumulate cluster centers
            vector_cluster_ids = simhash.get_cluster_ids(vectors)
            for cluster_id, (vec_idx, vec) in zip(vector_cluster_ids, enumerate(vectors)):
                cluster_centers[cluster_id] += vec
                cluster_center_id_to_vectors[cluster_id].append(vec_idx)

            if normalize_by_count or fill_empty_clusters:
                cluster_vector_counts = np.bincount(vector_cluster_ids, minlength=num_partitions)
                empty_mask = cluster_vector_counts == 0

            if normalize_by_count:
                assert empty_mask is not None
                assert cluster_vector_counts is not None
                non_empty_mask = ~empty_mask
                cluster_centers[non_empty_mask] /= cluster_vector_counts[non_empty_mask][:, None]

            # Fill empty clusters using vectors with minimum Hamming distance
            if fill_empty_clusters:
                assert empty_mask is not None
                assert precomputed_hamming_matrix is not None
                masked_hamming = np.where(
                    empty_mask[None, :], MAX_HAMMING_DISTANCE, precomputed_hamming_matrix
                )
                nearest_non_empty = np.argmin(masked_hamming, axis=1)
                fill_vectors = np.array(
                    [
                        vectors[cluster_center_id_to_vectors[cluster_id][0]]
                        for cluster_id in nearest_non_empty[empty_mask]
                    ]
                ).reshape(-1, self.dim)
                cluster_centers[empty_mask] = fill_vectors

            # Apply random projection for dimensionality reduction if needed
            if self.dim_proj < self.dim:
                dim_reduction_projection = self.dim_reduction_projections[
                    projection_index
                ]  # Get projection matrix for this repetition
                projected_centers = (1 / np.sqrt(self.dim_proj)) * (
                    cluster_centers @ dim_reduction_projection
                )

                # Flatten cluster centers into a single vector and add to output
                output_vectors.append(projected_centers.flatten())
                continue

            # If no projection needed (dim_proj == dim), use original cluster centers
            output_vectors.append(cluster_centers.flatten())

        # Concatenate results from all R_reps projections into final FDE
        return np.concatenate(output_vectors)


if __name__ == "__main__":
    v_arrs = np.random.randn(10, 100, 128)
    muvera = Muvera(128, 4, 8, 20, 42)

    for v_arr in v_arrs:
        muvera.process(v_arr)  # type: ignore
