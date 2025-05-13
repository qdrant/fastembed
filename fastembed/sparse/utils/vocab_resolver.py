from collections import defaultdict
from typing import Iterable

from py_rust_stemmers import SnowballStemmer
import numpy as np
from tokenizers import Tokenizer
from numpy.typing import NDArray

from fastembed.common.types import NumpyArray


class VocabTokenizerBase:
    def tokenize(self, sentence: str) -> NumpyArray:
        raise NotImplementedError()

    def convert_ids_to_tokens(self, token_ids: NumpyArray) -> list[str]:
        raise NotImplementedError()


class VocabTokenizer(VocabTokenizerBase):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, sentence: str) -> NumpyArray:
        return np.array(self.tokenizer.encode(sentence).ids)

    def convert_ids_to_tokens(self, token_ids: NumpyArray) -> list[str]:
        return [self.tokenizer.id_to_token(token_id) for token_id in token_ids]


class VocabResolver:
    def __init__(self, tokenizer: VocabTokenizerBase, stopwords: set[str], stemmer: SnowballStemmer):
        # Word to id mapping
        self.vocab: dict[str, int] = {}
        # Id to word mapping
        self.words: list[str] = []
        # Lemma to word mapping
        self.stem_mapping: dict[str, str] = {}
        self.tokenizer: VocabTokenizerBase = tokenizer
        self.stemmer = stemmer
        self.stopwords: set[str] = stopwords

    def tokenize(self, sentence: str) -> NumpyArray:
        return self.tokenizer.tokenize(sentence)

    def lookup_word(self, word_id: int) -> str:
        if word_id == 0:
            return "UNK"
        return self.words[word_id - 1]

    def convert_ids_to_tokens(self, token_ids: NumpyArray) -> list[str]:
        return self.tokenizer.convert_ids_to_tokens(token_ids)

    def vocab_size(self) -> int:
        # We need +1 for UNK token
        return len(self.vocab) + 1

    def save_vocab(self, path: str) -> None:
        with open(path, "w") as f:
            for word in self.words:
                f.write(word + "\n")

    def save_json_vocab(self, path: str) -> None:
        import json

        with open(path, "w") as f:
            json.dump({"vocab": self.words, "stem_mapping": self.stem_mapping}, f, indent=2)

    def load_json_vocab(self, path: str) -> None:
        import json

        with open(path, "r") as f:
            data = json.load(f)
            self.words = data["vocab"]
            self.vocab = {word: idx + 1 for idx, word in enumerate(self.words)}
            self.stem_mapping = data["stem_mapping"]

    def add_word(self, word: str) -> None:
        if word not in self.vocab:
            self.vocab[word] = len(self.vocab) + 1
            self.words.append(word)
            stem = self.stemmer.stem_word(word)
            if stem not in self.stem_mapping:
                self.stem_mapping[stem] = word
            else:
                existing_word = self.stem_mapping[stem]
                if len(existing_word) > len(word):
                    # Prefer shorter words for the same stem
                    # Example: "swim" is preferred over "swimming"
                    self.stem_mapping[stem] = word

    def load_vocab(self, path: str) -> None:
        with open(path, "r") as f:
            for line in f:
                self.add_word(line.strip())

    @classmethod
    def _reconstruct_bpe(
        cls, bpe_tokens: Iterable[tuple[int, str]]
    ) -> list[tuple[str, list[int]]]:
        result: list[tuple[str, list[int]]] = []
        acc: str = ""
        acc_idx: list[int] = []

        continuing_subword_prefix = "##"
        continuing_subword_prefix_len = len(continuing_subword_prefix)

        for idx, token in bpe_tokens:
            if token.startswith(continuing_subword_prefix):
                acc += token[continuing_subword_prefix_len:]
                acc_idx.append(idx)
            else:
                if acc:
                    result.append((acc, acc_idx))
                    acc_idx = []
                acc = token
                acc_idx.append(idx)

        if acc:
            result.append((acc, acc_idx))
        return result

    def resolve_tokens(
        self, token_ids: NDArray[np.int64]
    ) -> tuple[NDArray[np.int64], dict[int, int], dict[str, int], dict[str, list[str]]]:
        """
        Mark known tokens (including composed tokens) with vocab ids.

        Args:
            token_ids: (seq_len) - list of ids of tokens
                Example:
                    [
                        101,  3897, 19332, 12718, 23348,
                        1010,  1996,  7151,  2296, 4845,
                        2359,  2005,  4234,  1010,  4332,
                        2871,  3191,  2062, 102
                    ]

            returns:
                - token_ids with vocab ids
                    [
                        0,  151, 151, 0, 0,
                        912,  0,  0,  0, 332,
                        332,  332,  0,  7121,  191,
                        0,  0,  332, 0
                    ]
                - counts of each token
                    {
                        151: 1,
                        332: 3,
                        7121: 1,
                        191: 1,
                        912: 1
                    }
                - oov counts of each token
                    {
                        "the": 1,
                        "a": 1,
                        "[CLS]": 1,
                        "[SEP]": 1,
                        ...
                    }
                - forms of each token
                    {
                        "hello": ["hello"],
                        "world": ["worlds", "world", "worlding"],
                    }

        """
        tokens = self.convert_ids_to_tokens(token_ids)
        tokens_mapping = self._reconstruct_bpe(enumerate(tokens))

        counts: dict[int, int] = defaultdict(int)
        oov_count: dict[str, int] = defaultdict(int)

        forms: dict[str, list[str]] = defaultdict(list)

        for token, mapped_token_ids in tokens_mapping:
            vocab_id = 0
            if token in self.stopwords:
                vocab_id = 0
            elif token in self.vocab:
                vocab_id = self.vocab[token]
                forms[token].append(token)
            elif token in self.stem_mapping:
                vocab_id = self.vocab[self.stem_mapping[token]]
                forms[self.stem_mapping[token]].append(token)
            else:
                stem = self.stemmer.stem_word(token)
                if stem in self.stem_mapping:
                    vocab_id = self.vocab[self.stem_mapping[stem]]
                    forms[self.stem_mapping[stem]].append(token)

            for token_id in mapped_token_ids:
                token_ids[token_id] = vocab_id

            if vocab_id == 0:
                oov_count[token] += 1
            else:
                counts[vocab_id] += 1
        return token_ids, counts, oov_count, forms

