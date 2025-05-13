from typing import Dict, List, Set
from py_rust_stemmers import SnowballStemmer
from fastembed.common.utils import get_all_punctuation, remove_non_alphanumeric
import mmh3
import copy
from dataclasses import dataclass

import numpy as np

from fastembed.sparse.sparse_embedding_base import SparseEmbedding

GAP = 32000
INT32_MAX = 2**31 - 1


@dataclass
class WordEmbedding:
    word: str
    forms: List[str]
    count: int
    word_id: int
    embedding: List[float]


class SparseVectorConverter:
    def __init__(
        self,
        stopwords: Set[str],
        stemmer: SnowballStemmer,
        k: float = 1.2,
        b: float = 0.75,
        avg_len: float = 150.0,
    ):
        punctuation = set(get_all_punctuation())
        special_tokens = {"[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"}

        self.stemmer = stemmer
        self.unwanted_tokens = punctuation | special_tokens | stopwords

        self.k = k
        self.b = b
        self.avg_len = avg_len

    @classmethod
    def unkn_word_token_id(
        cls, word: str, shift: int
    ) -> int:  # 2-3 words can collide in 1 index with this mapping, not considering mm3 collisions
        token_hash = abs(mmh3.hash(word))

        range_size = INT32_MAX - shift
        remapped_hash = shift + (token_hash % range_size)

        return remapped_hash

    def bm25_tf(self, num_occurrences: int, sentence_len: int) -> float:
        res = num_occurrences * (self.k + 1)
        res /= num_occurrences + self.k * (1 - self.b + self.b * sentence_len / self.avg_len)
        return res

    @classmethod
    def normalize_vector(cls, vector: List[float]) -> List[float]:
        norm = sum([x**2 for x in vector]) ** 0.5
        if norm < 1e-8:
            return vector
        return [x / norm for x in vector]

    def clean_words(
        self, sentence_embedding: Dict[str, WordEmbedding], token_max_length: int = 40
    ) -> Dict[str, WordEmbedding]:
        """
        Clean miniCOIL-produced sentence_embedding, as unknown to the miniCOIL's stemmer tokens should fully resemble
        our BM25 token representation.

        sentence_embedding = {"9°": {"word": "9°", "word_id": -1, "count": 2, "embedding": [1], "forms": ["9°"]},
                "9": {"word": "9", "word_id": -1, "count": 2, "embedding": [1], "forms": ["9"]},
                "bat": {"word": "bat", "word_id": 2, "count": 3, "embedding": [0.2, 0.1, -0.2, -0.2], "forms": ["bats", "bat"]},
                "9°9": {"word": "9°9", "word_id": -1, "count": 1, "embedding": [1], "forms": ["9°9"]},
                "screech": {"word": "screech", "word_id": -1, "count": 1, "embedding": [1], "forms": ["screech"]},
                "screeched": {"word": "screeched", "word_id": -1, "count": 1, "embedding": [1], "forms": ["screeched"]}
                }
        cleaned_embedding_ground_truth = {
                "9": {"word": "9", "word_id": -1, "count": 6, "embedding": [1], "forms": ["9°", "9", "9°9", "9°9"]},
                "bat": {"word": "bat", "word_id": 2, "count": 3, "embedding": [0.2, 0.1, -0.2, -0.2], "forms": ["bats", "bat"]},
                "screech": {"word": "screech", "word_id": -1, "count": 2, "embedding": [1], "forms": ["screech", "screeched"]}
                }
        """

        new_sentence_embedding: Dict[str, WordEmbedding] = {}

        for word, embedding in sentence_embedding.items():
            # embedding = {
            #     "word": "vector",
            #     "forms": ["vector", "vectors"],
            #     "count": 2,
            #     "word_id": 1231,
            #     "embedding": [0.1, 0.2, 0.3, 0.4]
            # }
            if embedding.word_id > 0:
                # Known word, no need to clean
                new_sentence_embedding[word] = embedding
            else:
                # Unknown word
                if word in self.unwanted_tokens:
                    continue

                # Example complex word split:
                # word = `word^vec`
                word_cleaned = remove_non_alphanumeric(word).strip()
                # word_cleaned = `word vec`

                if len(word_cleaned) > 0:
                    # Subwords: ['word', 'vec']
                    for subword in word_cleaned.split():
                        stemmed_subword: str = self.stemmer.stem_word(subword)
                        if (
                            len(stemmed_subword) <= token_max_length
                            and stemmed_subword not in self.unwanted_tokens
                        ):
                            if stemmed_subword not in new_sentence_embedding:
                                new_sentence_embedding[stemmed_subword] = copy.deepcopy(embedding)
                                new_sentence_embedding[stemmed_subword].word = stemmed_subword
                            else:
                                new_sentence_embedding[stemmed_subword].count += embedding.count
                                new_sentence_embedding[stemmed_subword].forms += embedding.forms

        return new_sentence_embedding

    def embedding_to_vector(
        self,
        sentence_embedding: Dict[str, WordEmbedding],
        embedding_size: int,
        vocab_size: int,
    ) -> SparseEmbedding:
        """
        Convert miniCOIL sentence embedding to Qdrant sparse vector

        Example input:

        ```
        {
            "vector": WordEmbedding({ // Vocabulary word, encoded with miniCOIL normally
                "word": "vector",
                "forms": ["vector", "vectors"],
                "count": 2,
                "word_id": 1231,
                "embedding": [0.1, 0.2, 0.3, 0.4]
            }),
            "axiotic": WordEmbedding({ // Out-of-vocabulary word, fallback to BM25
                "word": "axiotic",
                "forms": ["axiotics"],
                "count": 1,
                "word_id": -1,
            })
        }
        ```

        """

        indices: List[int] = []
        values: List[float] = []
        
        # Example:
        # vocab_size = 10000
        # embedding_size = 4
        # GAP = 32000
        # 
        # We want to start random words section from the bucket, that is guaranteed to not
        # include any vocab words.
        # We need (vocab_size * embedding_size) slots for vocab words.
        # Therefore we need (vocab_size * embedding_size) // GAP + 1 buckets for vocab words.
        # Therefore, we can start random words from bucket (vocab_size * embedding_size) // GAP + 1 + 1

        # ID at which the scope of OOV words starts
        unknown_words_shift = (
            (vocab_size * embedding_size) // GAP + 2
        ) * GAP
        sentence_embedding_cleaned = self.clean_words(sentence_embedding)

        # Calculate sentence length after cleaning
        sentence_len = 0
        for embedding in sentence_embedding_cleaned.values():
            sentence_len += embedding.count

        for embedding in sentence_embedding_cleaned.values():
            word_id = embedding.word_id
            num_occurrences = embedding.count
            tf = self.bm25_tf(num_occurrences, sentence_len)
            if (
                word_id > 0
            ):  # miniCOIL starts with ID 1, we generally won't have word_id == 0 (UNK), as we don't add
                # these words to sentence_embedding
                embedding_values = embedding.embedding
                normalized_embedding = self.normalize_vector(embedding_values)

                for val_id, value in enumerate(normalized_embedding):
                    indices.append(
                        word_id * embedding_size + val_id
                    )  # since miniCOIL IDs start with 1
                    values.append(value * tf)
            else:
                indices.append(self.unkn_word_token_id(embedding.word, unknown_words_shift))
                values.append(tf)

        return SparseEmbedding(
            indices=np.array(indices, dtype=np.int32),
            values=np.array(values, dtype=np.float32),
        )

    def embedding_to_vector_query(
        self,
        sentence_embedding: Dict[str, WordEmbedding],
        embedding_size: int,
        vocab_size: int,
    ) -> SparseEmbedding:
        """
        Same as `embedding_to_vector`, but no TF
        """

        indices: List[int] = []
        values: List[float] = []

        # ID at which the scope of OOV words starts
        unknown_words_shift = ((vocab_size * embedding_size) // GAP + 2) * GAP

        sentence_embedding_cleaned = self.clean_words(sentence_embedding)

        for embedding in sentence_embedding_cleaned.values():
            word_id = embedding.word_id
            tf = 1.0

            if word_id >= 0:  # miniCOIL starts with ID 1
                embedding_values = embedding.embedding
                normalized_embedding = self.normalize_vector(embedding_values)

                for val_id, value in enumerate(normalized_embedding):
                    indices.append(
                        word_id * embedding_size + val_id
                    )  # since miniCOIL IDs start with 1
                    values.append(value * tf)
            else:
                indices.append(self.unkn_word_token_id(embedding.word, unknown_words_shift))
                values.append(tf)

        return SparseEmbedding(
            indices=np.array(indices, dtype=np.int32),
            values=np.array(values, dtype=np.float32),
        )
