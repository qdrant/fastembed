# This code is a modified copy of the `NLTKWordTokenizer` class from `NLTK` library.

import re


class SimpleTokenizer:
    @staticmethod
    def tokenize(text: str) -> list[str]:
        text = re.sub(r"[^\w]", " ", text.lower())
        text = re.sub(r"\s+", " ", text)

        return text.strip().split()


class WordTokenizer:
    """The tokenizer is "destructive" such that the regexes applied will munge the
    input string to a state beyond re-construction.
    """

    # Starting quotes.
    STARTING_QUOTES = [
        (re.compile("([«“‘„]|[`]+)", re.U), r" \1 "),
        (re.compile(r"^\""), r"``"),
        (re.compile(r"(``)"), r" \1 "),
        (re.compile(r"([ \(\[{<])(\"|\'{2})"), r"\1 `` "),
        (re.compile(r"(?i)(\')(?!re|ve|ll|m|t|s|d|n)(\w)\b", re.U), r"\1 \2"),
    ]

    # Ending quotes.
    ENDING_QUOTES = [
        (re.compile("([»”’])", re.U), r" \1 "),
        (re.compile(r"''"), " '' "),
        (re.compile(r'"'), " '' "),
        (re.compile(r"([^' ])('[sS]|'[mM]|'[dD]|') "), r"\1 \2 "),
        (re.compile(r"([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) "), r"\1 \2 "),
    ]

    # Punctuation.
    PUNCTUATION = [
        (re.compile(r'([^\.])(\.)([\]\)}>"\'' "»”’ " r"]*)\s*$", re.U), r"\1 \2 \3 "),
        (re.compile(r"([:,])([^\d])"), r" \1 \2"),
        (re.compile(r"([:,])$"), r" \1 "),
        (
            re.compile(r"\.{2,}", re.U),
            r" \g<0> ",
        ),
        (re.compile(r"[;@#$%&]"), r" \g<0> "),
        (
            re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'),
            r"\1 \2\3 ",
        ),  # Handles the final period.
        (re.compile(r"[?!]"), r" \g<0> "),
        (re.compile(r"([^'])' "), r"\1 ' "),
        (
            re.compile(r"[*]", re.U),
            r" \g<0> ",
        ),
    ]

    # Pads parentheses
    PARENS_BRACKETS = (re.compile(r"[\]\[\(\)\{\}\<\>]"), r" \g<0> ")
    DOUBLE_DASHES = (re.compile(r"--"), r" -- ")

    # List of contractions adapted from Robert MacIntyre's tokenizer.
    CONTRACTIONS2 = [
        re.compile(pattern)
        for pattern in (
            r"(?i)\b(can)(?#X)(not)\b",
            r"(?i)\b(d)(?#X)('ye)\b",
            r"(?i)\b(gim)(?#X)(me)\b",
            r"(?i)\b(gon)(?#X)(na)\b",
            r"(?i)\b(got)(?#X)(ta)\b",
            r"(?i)\b(lem)(?#X)(me)\b",
            r"(?i)\b(more)(?#X)('n)\b",
            r"(?i)\b(wan)(?#X)(na)(?=\s)",
        )
    ]
    CONTRACTIONS3 = [
        re.compile(pattern) for pattern in (r"(?i) ('t)(?#X)(is)\b", r"(?i) ('t)(?#X)(was)\b")
    ]

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        """Return a tokenized copy of `text`.

        >>> s = '''Good muffins cost $3.88 (roughly 3,36 euros)\nin New York.'''
        >>> WordTokenizer().tokenize(s)
        ['Good', 'muffins', 'cost', '$', '3.88', '(', 'roughly', '3,36', 'euros', ')', 'in', 'New', 'York', '.']

        Args:
            text: The text to be tokenized.

        Returns:
            A list of tokens.
        """
        for regexp, substitution in cls.STARTING_QUOTES:
            text = regexp.sub(substitution, text)

        for regexp, substitution in cls.PUNCTUATION:
            text = regexp.sub(substitution, text)

        # Handles parentheses.
        regexp, substitution = cls.PARENS_BRACKETS
        text = regexp.sub(substitution, text)

        # Handles double dash.
        regexp, substitution = cls.DOUBLE_DASHES
        text = regexp.sub(substitution, text)

        # add extra space to make things easier
        text = " " + text + " "

        for regexp, substitution in cls.ENDING_QUOTES:
            text = regexp.sub(substitution, text)

        for regexp in cls.CONTRACTIONS2:
            text = regexp.sub(r" \1 \2 ", text)
        for regexp in cls.CONTRACTIONS3:
            text = regexp.sub(r" \1 \2 ", text)
        return text.split()
