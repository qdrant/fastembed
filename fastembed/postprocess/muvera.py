import numpy as np

from fastembed.common.types import NumpyArray


class SimHashProjection:
    """
    SimHash projection component for MUVERA clustering.

    This class implements locality-sensitive hashing using random hyperplanes
    to partition the vector space into 2^k_sim clusters. Each vector is assigned
    to a cluster based on which side of k_sim random hyperplanes it falls on.

    Attributes:
        k_sim (int): Number of SimHash functions (hyperplanes)
        d (int): Dimensionality of input vectors
        simhash_vectors (np.ndarray): Random hyperplane normal vectors of shape (d, k_sim)
    """

    def __init__(self, k_sim: int, d: int, random_generator: np.random.Generator):
        """
        Initialize SimHash projection with random hyperplanes.

        Args:
            k_sim (int): Number of SimHash functions, determines 2^k_sim clusters
            d (int): Dimensionality of input vectors
            random_generator (np.random.Generator): Random number generator for reproducibility
        """
        self.k_sim = k_sim
        self.d = d
        # Generate k_sim random hyperplanes (normal vectors) from standard normal distribution
        self.simhash_vectors = random_generator.normal(size=(d, k_sim))

    def get_cluster_id(self, vector: np.ndarray) -> int:
        """
        Compute the cluster ID for a given vector using SimHash.

        The cluster ID is determined by computing the dot product of the vector
        with each hyperplane normal vector, taking the sign, and interpreting
        the resulting binary string as an integer.

        Args:
            vector (np.ndarray): Input vector of shape (d,)

        Returns:
            int: Cluster ID in range [0, 2^k_sim - 1]

        Raises:
            AssertionError: If a vector shape doesn't match expected dimensionality
        """
        assert vector.shape == (
            self.d,
        ), f"Expected vector of shape ({self.d},), got {vector.shape}"

        # Project vector onto each hyperplane normal vector
        dot_product = np.dot(vector, self.simhash_vectors)

        # Apply sign function to get binary values (1 if positive, 0 if negative)
        binary_values = (dot_product > 0).astype(int)

        # Convert binary representation to decimal cluster ID
        # Each bit position i contributes bit_value * 2^i to the final ID
        cluster_id = 0
        for i, bit in enumerate(binary_values):
            cluster_id += bit * (2**i)
        return cluster_id


class MuveraPostprocessor:
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
        d (int): Input vector dimensionality
        d_proj (int): Output dimensionality after random projection
        R_reps (int): Number of random projection repetitions
        simhash_projections (List[SimHashProjection]): SimHash instances for clustering
        S_projections (np.ndarray): Random projection matrices of shape (R_reps, d, d_proj)
    """

    def __init__(
        self,
        k_sim: int,
        d: int,
        d_proj: int,
        R_reps: int,
        random_seed: np.random.Generator = 42,
    ):
        """
        Initialize MUVERA algorithm with specified parameters.

        Args:
            k_sim (int): Number of SimHash functions (creates 2^k_sim clusters)
            d (int): Dimensionality of input vectors
            d_proj (int): Dimensionality after random projection (must be <= d)
            R_reps (int): Number of random projection repetitions for robustness
            random_seed (np.random.Generator): Seed for random generator

        Raises:
            ValueError: If d_proj > d (cannot project to higher dimensionality)
        """
        if d_proj > d:
            raise ValueError(
                f"Cannot project to a higher dimensionality (d_proj={d_proj} > d={d})"
            )

        self.k_sim = k_sim
        self.d = d
        self.d_proj = d_proj
        self.R_reps = R_reps
        # Create R_reps independent SimHash projections for robustness
        generator = np.random.default_rng(random_seed)
        self.simhash_projections = [
            SimHashProjection(k_sim=self.k_sim, d=self.d, random_generator=generator)
            for _ in range(R_reps)
        ]
        # Random projection matrices with entries from {-1, +1} for each repetition
        self.S_projections = random_seed.choice([-1, 1], size=(R_reps, d, d_proj))

    def get_output_dimension(self) -> int:
        """
        Get the output dimension of the MUVERA algorithm.

        Returns:
            int: Output dimension (R_reps * B * d_proj) where B = 2^k_sim
        """
        B = 2**self.k_sim
        return self.R_reps * B * self.d_proj

    def process_document(self, vectors: np.ndarray) -> np.ndarray:
        """
        Encode a document's vectors into a Fixed Dimensional Encoding (FDE).

        Uses document-specific settings: normalizes cluster centers by vector count
        and fills empty clusters using Hamming distance-based selection.

        Args:
            vectors (np.ndarray): Document vectors of shape (n_tokens, d)

        Returns:
            np.ndarray: Fixed dimensional encoding of shape (R_reps * B * d_proj,)
        """
        return self.process(vectors, fill_empty_clusters=True, normalize_by_count=True)

    def process_query(self, vectors: np.ndarray) -> np.ndarray:
        """
        Encode a query's vectors into a Fixed Dimensional Encoding (FDE).

        Uses query-specific settings: no normalization by count and no empty
        cluster filling to preserve query vector magnitudes.

        Args:
            vectors (np.ndarray): Query vectors of shape (n_tokens, d)

        Returns:
            np.ndarray: Fixed dimensional encoding of shape (R_reps * B * d_proj,)
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
        1. For each of R_reps random projections:
           a. Assign vectors to clusters using SimHash
           b. Compute cluster centers (sum of vectors in each cluster)
           c. Optionally normalize by cluster size
           d. Fill empty clusters using Hamming distance if requested
           e. Apply random projection for dimensionality reduction
           f. Flatten cluster centers into a vector
        2. Concatenate all projection results

        Args:
            vectors (np.ndarray): Input vectors of shape (n_vectors, d)
            fill_empty_clusters (bool): Whether to fill empty clusters using nearest
                                      vectors based on Hamming distance of cluster IDs
            normalize_by_count (bool): Whether to normalize cluster centers by the
                                     number of vectors assigned to each cluster

        Returns:
            np.ndarray: Fixed dimensional encoding of shape (R_reps * B * d_proj)
                        where B = 2^k_sim is the number of clusters

        Raises:
            AssertionError: If input vectors don't have expected dimensionality
        """
        assert (
            vectors.shape[1] == self.d
        ), f"Expected vectors of shape (n, {self.d}), got {vectors.shape}"

        # Store results from each random projection
        output_vectors = []

        # B is the number of clusters (2^k_sim)
        B = 2**self.k_sim
        for projection_index, simhash in enumerate(self.simhash_projections):
            # Initialize cluster centers and count vectors assigned to each cluster
            cluster_centers = np.zeros((B, self.d))
            cluster_vector_counts = np.zeros(B)

            # Assign each vector to its cluster and accumulate cluster centers
            for vector in vectors:
                cluster_id = simhash.get_cluster_id(vector)
                cluster_centers[cluster_id] += vector
                cluster_vector_counts[cluster_id] += 1

            # Normalize cluster centers by the number of vectors (for documents)
            if normalize_by_count:
                for i in range(B):
                    if cluster_vector_counts[i] == 0:
                        continue  # Skip empty clusters
                    cluster_centers[i] /= cluster_vector_counts[i]

            # Fill empty clusters using vectors with minimum Hamming distance
            if fill_empty_clusters:
                for i in range(B):
                    if cluster_vector_counts[i] == 0:  # Empty cluster found
                        min_hamming = float("inf")
                        best_vector = None
                        # Find vector whose cluster ID has minimum Hamming distance to i
                        for vector in vectors:
                            vector_cluster_id = simhash.get_cluster_id(vector)
                            # Hamming distance = number of differing bits in binary representation
                            hamming_dist = bin(i ^ vector_cluster_id).count("1")
                            if hamming_dist < min_hamming:
                                min_hamming = hamming_dist
                                best_vector = vector
                        # Assign the best matching vector to the empty cluster
                        if best_vector is not None:
                            cluster_centers[i] = best_vector

            # Apply random projection for dimensionality reduction if needed
            if self.d_proj < self.d:
                S = self.S_projections[
                    projection_index
                ]  # Get projection matrix for this repetition
                projected_centers = (1 / np.sqrt(self.d_proj)) * np.dot(cluster_centers, S)

                # Flatten cluster centers into a single vector and add to output
                output_vectors.append(projected_centers.flatten())
                continue

            # If no projection needed (d_proj == d), use original cluster centers
            output_vectors.append(cluster_centers.flatten())

        # Concatenate results from all R_reps projections into final FDE
        return np.concatenate(output_vectors)
