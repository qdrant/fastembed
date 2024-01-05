from typing import Callable, Optional, List
from pydantic import BaseModel, model_validator
from tokenizers import Tokenizer

"""
For how to use validators in Pydantic, see: https://docs.pydantic.dev/latest/concepts/validators/
"""


class TextSplitterConfig(BaseModel):
    """
    Configuration for TextSplitter.

    Attributes:
        chunk_size (int): Size of the chunks, measured in tokens.
        chunk_overlap (int): Overlap between chunks, measured in tokens.
        length_function (Callable[[str], int]): Function that measures the length of given chunks. Default is len.
        keep_separator (bool): Whether to keep the separator or not. Default is False.
        strip_whitespace (bool): Whether to strip whitespace or not. Default is True.
        tokenizer (Optional[Tokenizer]): Tokenizer used in FastEmbedRecursiveSplitter. Default is None.
        is_separator_regex (bool): Whether the separator is a regular expression. Default is False.
        separators (Optional[List[str]]): List of separators. Default is None.
    """

    chunk_size: int
    chunk_overlap: int
    length_function: Callable[[str], int] = len
    keep_separator: bool = False
    strip_whitespace: bool = True
    tokenizer: Optional[Tokenizer] = None
    is_separator_regex: bool = False
    separators: Optional[List[str]] = None

    @model_validator(mode="after")
    def _check_chunk_size_overlap(self) -> "TextSplitterConfig":
        """
        Validates chunk_size and chunk_overlap.

        Raises:
            ValueError: If chunk_size is not greater than 0, chunk_overlap is negative, or chunk_overlap is greater than chunk_size.
        """
        chunk_size = self.chunk_size
        chunk_overlap = self.chunk_overlap
        if chunk_size <= 0:
            raise ValueError("Invalid value for chunk_size. It must be greater than 0.")
        if chunk_overlap < 0:
            raise ValueError("Invalid value for chunk_overlap. It must be greater than or equal to 0.")
        if chunk_overlap > chunk_size:
            raise ValueError("Invalid value for chunk_overlap. It must be smaller than chunk_size.")
        return self

    @model_validator(mode="after")
    def _check_tokenizer(self) -> "TextSplitterConfig":
        """
        Validates tokenizer.

        Raises:
            ValueError: If tokenizer is not an instance of Tokenizer or its max_length is not greater than or equal to chunk_size.
        """
        tokenizer = self.tokenizer
        chunk_size = self.chunk_size
        if tokenizer is None:
            return self
        if not isinstance(tokenizer, Tokenizer):
            raise ValueError("Invalid tokenizer. It must be an instance of tokenizers.Tokenizer.")
        if tokenizer.model_max_length < chunk_size:
            raise ValueError("Invalid chunk size. It must be smaller than or equal to tokenizer's max_length.")
        return self
