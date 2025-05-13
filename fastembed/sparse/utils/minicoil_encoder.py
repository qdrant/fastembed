"""
Pure numpy implementation of encoder model for a single word.

This model is not trainable, and should only be used for inference.
"""

import numpy as np
from fastembed.common.types import NumpyArray


class Encoder:
    """
    Encoder(768, 4, 10000)

    Will look like this:


                                         Per-word
                                         Encoder Matrix
     ┌─────────────────────┐
     │ Token Embedding(768)├──────┐      (10k, 768, 4)
     └─────────────────────┘      │         ┌─────────┐
                                  │         │         │
     ┌─────────────────────┐      │       ┌─┴───────┐ │
     │                     │      │       │         │ │
     └─────────────────────┘      │     ┌─┴───────┐ │ │      ┌─────────┐
                                  └────►│         │ │ ├─────►│Tanh     │
     ┌─────────────────────┐            │         │ │ │      └─────────┘
     │                     │            │         │ ├─┘
     └─────────────────────┘            │         ├─┘
                                        │         │
     ┌─────────────────────┐            └─────────┘
     │                     │
     └─────────────────────┘

     Final linear transformation is accompanied by a non-linear activation function: Tanh.

     Tanh is used to ensure that the output is in the range [-1, 1].
     It would be easier to visually interpret the output of the model, assuming that each dimension
     would need to encode a type of semantic cluster.
    """

    def __init__(
        self,
        weights: NumpyArray,
    ):
        self.weights = weights
        self.vocab_size, self.input_dim, self.output_dim = weights.shape

        self.encoder_weights: NumpyArray = weights

        # Activation function
        self.activation = np.tanh

    @staticmethod
    def convert_vocab_ids(vocab_ids: NumpyArray) -> NumpyArray:
        """
        Convert vocab_ids of shape (batch_size, seq_len) into (batch_size, seq_len, 2)
        by appending batch_id alongside each vocab_id.
        """
        batch_size, seq_len = vocab_ids.shape
        batch_ids = np.arange(batch_size, dtype=vocab_ids.dtype).reshape(batch_size, 1)
        batch_ids = np.repeat(batch_ids, seq_len, axis=1)
        # Stack vocab_ids and batch_ids along the last dimension
        combined: NumpyArray = np.stack((vocab_ids, batch_ids), axis=2).astype(np.int32)
        return combined

    @classmethod
    def avg_by_vocab_ids(
        cls, vocab_ids: NumpyArray, embeddings: NumpyArray
    ) -> tuple[NumpyArray, NumpyArray]:
        """
        Takes:
            vocab_ids: (batch_size, seq_len) int array
            embeddings: (batch_size, seq_len, input_dim) float array

        Returns:
            unique_flattened_vocab_ids: (total_unique, 2) array of [vocab_id, batch_id]
            unique_flattened_embeddings: (total_unique, input_dim) averaged embeddings
        """
        input_dim = embeddings.shape[2]

        # Flatten vocab_ids and embeddings
        # flattened_vocab_ids: (batch_size*seq_len, 2)
        flattened_vocab_ids = cls.convert_vocab_ids(vocab_ids).reshape(-1, 2)

        # flattened_embeddings: (batch_size*seq_len, input_dim)
        flattened_embeddings = embeddings.reshape(-1, input_dim)

        # Find unique (vocab_id, batch_id) pairs
        unique_flattened_vocab_ids, inverse_indices = np.unique(
            flattened_vocab_ids, axis=0, return_inverse=True
        )

        # Prepare arrays to accumulate sums
        unique_count = unique_flattened_vocab_ids.shape[0]
        unique_flattened_embeddings = np.zeros((unique_count, input_dim), dtype=np.float32)
        unique_flattened_count = np.zeros(unique_count, dtype=np.int32)

        # Use np.add.at to accumulate sums based on inverse indices
        np.add.at(unique_flattened_embeddings, inverse_indices, flattened_embeddings)
        np.add.at(unique_flattened_count, inverse_indices, 1)

        # Compute averages
        unique_flattened_embeddings /= unique_flattened_count[:, None]

        return unique_flattened_vocab_ids.astype(np.int32), unique_flattened_embeddings.astype(
            np.float32
        )

    def forward(
        self, vocab_ids: NumpyArray, embeddings: NumpyArray
    ) -> tuple[NumpyArray, NumpyArray]:
        """
        Args:
            vocab_ids: (batch_size, seq_len) int array
            embeddings: (batch_size, seq_len, input_dim) float array

        Returns:
            unique_flattened_vocab_ids_and_batch_ids: (total_unique, 2)
            unique_flattened_encoded: (total_unique, output_dim)
        """
        # Average embeddings for duplicate vocab_ids
        unique_flattened_vocab_ids_and_batch_ids, unique_flattened_embeddings = (
            self.avg_by_vocab_ids(vocab_ids, embeddings)
        )

        # Select the encoder weights for each unique vocab_id
        unique_flattened_vocab_ids = unique_flattened_vocab_ids_and_batch_ids[:, 0].astype(
            np.int32
        )

        # unique_encoder_weights: (total_unique, input_dim, output_dim)
        unique_encoder_weights = self.encoder_weights[unique_flattened_vocab_ids]

        # Compute linear transform: (total_unique, output_dim)
        # Using Einstein summation for matrix multiplication:
        # 'bi,bio->bo' means: for each "b" (batch element), multiply embeddings (b,i) by weights (b,i,o) -> (b,o)
        unique_flattened_encoded = np.einsum(
            "bi,bio->bo", unique_flattened_embeddings, unique_encoder_weights
        )

        # Apply Tanh activation and ensure float32 type
        unique_flattened_encoded = self.activation(unique_flattened_encoded).astype(np.float32)

        return unique_flattened_vocab_ids_and_batch_ids.astype(np.int32), unique_flattened_encoded
