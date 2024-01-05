# Custom implementation based on the Langchain text splitter
# Provided under standard MIT license by LangChain
# https://github.com/hwchase17/langchain

import logging
import re
from abc import ABC, abstractmethod
from typing import (
    Iterable,
    List,
    Optional,
)

from tokenizers import Tokenizer

from .models import TextSplitterConfig

logger = logging.getLogger(__name__)


def _split_text_with_regex(text: str, separator: str, keep_separator: bool) -> List[str]:
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
            if len(_splits) % 2 == 0:
                splits += _splits[-1:]
            splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class TextSplitter(ABC):
    """Interface for splitting text into chunks."""

    def __init__(
        self,
        config: TextSplitterConfig,
    ) -> None:
        """Create a new TextSplitter.

        Args:
            chunk_size: Maximum token length per chunk
            chunk_overlap: Overlap in tokens between chunks
            length_function: Function that measures the length of given chunks
            keep_separator: Whether to keep the separator in the chunks
            strip_whitespace: If `True`, strips whitespace from the start and end of
                              every document
        """
        self._config = config
        self._chunk_size = config.chunk_size
        self._chunk_overlap = config.chunk_overlap
        self._length_function = config.length_function
        self._keep_separator = config.keep_separator
        self._strip_whitespace = config.strip_whitespace

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into multiple components."""

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        text = separator.join(docs)
        if self._strip_whitespace:
            text = text.strip()
        if text == "":
            return None
        else:
            return text

    def _merge_splits(
        self,
        splits: Iterable[str],
        separator: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[str]:
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        separator_len = self._length_function(separator)

        chunk_size = chunk_size or self._chunk_size
        chunk_overlap = chunk_overlap or self._chunk_overlap

        docs = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            _len = self._length_function(d)
            if total + _len + (separator_len if len(current_doc) > 0 else 0) > chunk_size:
                if total > chunk_size:
                    logger.warning(
                        f"Created a chunk of size {total}, " f"which is longer than the specified {chunk_size}"
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0) > chunk_size and total > 0
                    ):
                        total -= self._length_function(current_doc[0]) + (separator_len if len(current_doc) > 1 else 0)
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs


class FastEmbedRecursiveSplitter(TextSplitter):
    """
    Splitting text into chunks recursively.

    The splitter splits text into chunks of a maximum size, with a given overlap.
    """

    def __init__(
        self,
        config: TextSplitterConfig,
    ) -> None:
        """Create a new TextSplitter."""
        tokenizer = config.tokenizer
        if not isinstance(tokenizer, Tokenizer):
            raise ValueError("Tokenizer received was not an instance of tokenizers.Tokenizer")

        def _tokenizer_length(text: str) -> int:
            return len(tokenizer.encode(text))

        config.length_function = _tokenizer_length
        super().__init__(config)
        self._separators = config.separators or ["\n\n", "\n", " ", ""]
        self._is_separator_regex = config.is_separator_regex

    def _split_text(
        self, text: str, separators: List[str], chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None
    ) -> List[str]:
        """Split incoming text and return chunks."""

        chunk_size = chunk_size or self._chunk_size
        chunk_overlap = chunk_overlap or self._chunk_overlap

        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1 :]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(text, _separator, self._keep_separator)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator, chunk_size, chunk_overlap)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return final_chunks

    def split_text(self, text: str, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> List[str]:
        return self._split_text(text, self._separators, chunk_size, chunk_overlap)
