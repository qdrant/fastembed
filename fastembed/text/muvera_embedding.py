import numpy as np
from typing import Any, Iterable, Optional, Union
from dataclasses import asdict

from fastembed.common.types import NumpyArray
from fastembed.common.model_description import DenseModelDescription
from fastembed.text.text_embedding_base import TextEmbeddingBase


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
            random_generator (np.random.Generator): Random number random_generator for reproducibility
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
            AssertionError: If vector shape doesn't match expected dimensionality
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


class MuveraAlgorithm:
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
        random_generator: np.random.Generator,
    ):
        """
        Initialize MUVERA algorithm with specified parameters.

        Args:
            k_sim (int): Number of SimHash functions (creates 2^k_sim clusters)
            d (int): Dimensionality of input vectors
            d_proj (int): Dimensionality after random projection (must be <= d)
            R_reps (int): Number of random projection repetitions for robustness
            random_generator (np.random.Generator): Random number random_generator for reproducibility

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
        self.simhash_projections = [
            SimHashProjection(k_sim=self.k_sim, d=self.d, random_generator=random_generator)
            for _ in range(R_reps)
        ]
        # Random projection matrices with entries from {-1, +1} for each repetition
        self.S_projections = random_generator.choice([-1, 1], size=(R_reps, d, d_proj))

    def get_output_dimension(self) -> int:
        """
        Get the output dimension of the MUVERA algorithm.

        Returns:
            int: Output dimension (R_reps * B * d_proj) where B = 2^k_sim
        """
        B = 2**self.k_sim
        return self.R_reps * B * self.d_proj

    def encode_document(self, vectors: np.ndarray) -> np.ndarray:
        """
        Encode a document's vectors into a Fixed Dimensional Encoding (FDE).

        Uses document-specific settings: normalizes cluster centers by vector count
        and fills empty clusters using Hamming distance-based selection.

        Args:
            vectors (np.ndarray): Document vectors of shape (n_tokens, d)

        Returns:
            np.ndarray: Fixed dimensional encoding of shape (R_reps * B * d_proj,)
        """
        return self.encode(vectors, fill_empty_clusters=True, normalize_by_count=True)

    def encode_query(self, vectors: np.ndarray) -> np.ndarray:
        """
        Encode a query's vectors into a Fixed Dimensional Encoding (FDE).

        Uses query-specific settings: no normalization by count and no empty
        cluster filling to preserve query vector magnitudes.

        Args:
            vectors (np.ndarray): Query vectors of shape (n_tokens, d)

        Returns:
            np.ndarray: Fixed dimensional encoding of shape (R_reps * B * d_proj,)
        """
        return self.encode(vectors, fill_empty_clusters=False, normalize_by_count=False)

    def encode(
        self,
        vectors: np.ndarray,
        fill_empty_clusters: bool = True,
        normalize_by_count: bool = True,
    ) -> np.ndarray:
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
            np.ndarray: Fixed dimensional encoding of shape (R_reps * B * d_proj,)
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


class MuveraEmbedding(TextEmbeddingBase):
    """
    MUVERA (Multi-Vector Retrieval Architecture) text embedding class.

    This class combines late interaction models (like ColBERT) with the MUVERA algorithm
    to create fixed-dimensional embeddings from variable-length token sequences.
    It's compatible with the fastembed TextEmbedding interface while supporting
    multivector capabilities through late interaction models.

    The MUVERA algorithm transforms variable-length token embeddings into fixed-dimensional
    embeddings using SimHash clustering and random projections, making it suitable for
    traditional dense retrieval systems while preserving the benefits of late interaction.

    Attributes:
        late_interaction_model (LateInteractionTextEmbedding): The underlying late interaction model
        muvera_algorithm (MuveraAlgorithm): The MUVERA algorithm instance
        k_sim (int): Number of SimHash functions (controls clustering)
        d_proj (int): Dimensionality after random projection
        R_reps (int): Number of random projection repetitions
    """

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        k_sim: int = 4,
        d_proj: int = 32,
        R_reps: int = 10,
        random_seed: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Initialize MuveraEmbedding with a late interaction model and MUVERA parameters.

        Args:
            model_name (str): Name of the late interaction model to use.
                             Must be a supported late interaction model (ColBERT, JinaColBERT, etc.)
            cache_dir (Optional[str]): Cache directory for model files
            threads (Optional[int]): Number of threads for model inference
            k_sim (int): Number of SimHash functions (creates 2^k_sim clusters). Default: 4
            d_proj (int): Dimensionality after random projection. Default: 32
            R_reps (int): Number of random projection repetitions for robustness. Default: 10
            random_seed (Optional[int]): Random seed for reproducibility. Default: None
            **kwargs: Additional arguments passed to the late interaction model

        Raises:
            ValueError: If the model_name is not a supported late interaction model
        """
        super().__init__(model_name, cache_dir, threads, **kwargs)

        # Initialize the late interaction model (import locally to avoid circular imports)
        try:
            from fastembed.late_interaction.late_interaction_text_embedding import (
                LateInteractionTextEmbedding,
            )

            self.late_interaction_model = LateInteractionTextEmbedding(
                model_name=model_name, cache_dir=cache_dir, threads=threads, **kwargs
            )
        except ValueError as e:
            raise ValueError(
                f"Model {model_name} is not supported as a late interaction model. "
                f"Please use a supported late interaction model like 'colbert-ir/colbertv2.0' or 'jinaai/jina-colbert-v2'. "
                f"Original error: {e}"
            )

        # Store MUVERA parameters
        self.k_sim = k_sim
        self.d_proj = d_proj
        self.R_reps = R_reps

        # Get the token embedding dimension from the late interaction model
        self.token_dim = self.late_interaction_model.embedding_size

        # Initialize MUVERA algorithm
        generator = np.random.default_rng(random_seed)
        self.muvera_algorithm = MuveraAlgorithm(
            k_sim=k_sim,
            d=self.token_dim,
            d_proj=d_proj,
            R_reps=R_reps,
            random_generator=generator,
        )

        # Cache the output embedding size
        self._embedding_size: int = self.muvera_algorithm.get_output_dimension()

    @property
    def embedding_size(self) -> int:
        """Get the embedding size of the MUVERA output."""
        return self._embedding_size

    @classmethod
    def get_embedding_size(
        cls, model_name: str, k_sim: int = 4, d_proj: int = 32, R_reps: int = 10
    ) -> int:
        """
        Get the embedding size for a given model and MUVERA parameters.

        Args:
            model_name (str): Name of the late interaction model
            k_sim (int): Number of SimHash functions. Default: 4
            d_proj (int): Dimensionality after random projection. Default: 32
            R_reps (int): Number of random projection repetitions. Default: 10

        Returns:
            int: The size of the MUVERA embedding (R_reps * 2^k_sim * d_proj)

        Raises:
            ValueError: If the model name is not found in supported models
        """
        # Calculate MUVERA output dimension
        B = 2**k_sim
        return R_reps * B * d_proj

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        """
        List supported models (same as late interaction models).

        Returns:
            list[DenseModelDescription]: List of supported late interaction models
        """
        from fastembed.late_interaction.late_interaction_text_embedding import (
            LateInteractionTextEmbedding,
        )

        return LateInteractionTextEmbedding._list_supported_models()

    @classmethod
    def list_supported_models(cls) -> list[dict[str, Any]]:
        """
        Lists the supported models.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing the model information.
        """
        return [asdict(model) for model in cls._list_supported_models()]

    def embed(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        """
        Encode a list of documents into MUVERA embeddings.

        This method uses the late interaction model to get token-level embeddings,
        then applies the MUVERA algorithm to create fixed-dimensional embeddings.
        Documents are encoded with normalization and empty cluster filling.

        Args:
            documents: Iterator of documents or single document to embed
            batch_size: Batch size for encoding -- higher values will use more memory, but be faster
            parallel: If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
                     If 0, use all available cores.
                     If None, don't use data-parallel processing, use default onnxruntime threading instead.

        Returns:
            Iterable[NumpyArray]: List of MUVERA embeddings, one per document
        """
        # Handle single string input
        if isinstance(documents, str):
            documents = [documents]

        # Get token-level embeddings from the late interaction model
        token_embeddings = self.late_interaction_model.embed(
            documents=documents, batch_size=batch_size, parallel=parallel, **kwargs
        )

        # Apply MUVERA algorithm to each document's token embeddings
        for token_embedding in token_embeddings:
            # token_embedding shape: (n_tokens, token_dim)
            muvera_embedding = self.muvera_algorithm.encode_document(token_embedding)
            yield muvera_embedding.astype(np.float32)

    def query_embed(self, query: Union[str, Iterable[str]], **kwargs: Any) -> Iterable[NumpyArray]:
        """
        Embeds queries using MUVERA algorithm.

        This method uses the late interaction model to get token-level embeddings,
        then applies the MUVERA algorithm with query-specific settings (no normalization
        and no empty cluster filling) to preserve query vector magnitudes.

        Args:
            query (Union[str, Iterable[str]]): The query to embed, or an iterable e.g. list of queries.

        Returns:
            Iterable[NumpyArray]: The MUVERA query embeddings.
        """
        # Handle single string input
        if isinstance(query, str):
            query = [query]

        # Get token-level embeddings from the late interaction model
        token_embeddings = self.late_interaction_model.query_embed(query, **kwargs)

        # Apply MUVERA algorithm to each query's token embeddings
        for token_embedding in token_embeddings:
            # token_embedding shape: (n_tokens, token_dim)
            muvera_embedding = self.muvera_algorithm.encode_query(token_embedding)
            yield muvera_embedding.astype(np.float32)

    def passage_embed(self, texts: Iterable[str], **kwargs: Any) -> Iterable[NumpyArray]:
        """
        Embeds a list of text passages into MUVERA embeddings.

        This is an alias for the embed method, following the fastembed interface.

        Args:
            texts (Iterable[str]): The list of texts to embed.
            **kwargs: Additional keyword arguments to pass to the embed method.

        Yields:
            Iterable[NumpyArray]: The MUVERA embeddings.
        """
        yield from self.embed(texts, **kwargs)
