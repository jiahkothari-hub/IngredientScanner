"""Word-level tokenisation with character offsets.

Why our own tokenizer? Person 2's Transformer models split words into sub-words themselves.
They need WORD-level tokens plus one label per word (Hugging Face `is_split_into_words=True`).
Character offsets let us convert character-level entity spans <-> word-level BIO tags exactly.

Rules (tried in this order at each position):
  1. a number with a percent sign          "20.7%"   (one token: percentages are not entities)
  2. a run of letters/digits (+ apostrophe) "E330", "150d", "B12", "baker's", "Maida"
  3. any other single non-space character  "(", ",", "&", "%", "-"

Examples:
  "INS 330"   -> ["INS", "330"]
  "503(ii)"   -> ["503", "(", "ii", ")"]
  "mono- and diglycerides" -> ["mono", "-", "and", "diglycerides"]
"""
import re

TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)?%|[^\W_]+(?:'[^\W_]+)*|\S")


def tokenize_with_offsets(text: str) -> list:
    """Return [(token, start, end), ...] where text[start:end] == token."""
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def tokenize(text: str) -> list:
    return [token for token, _, _ in tokenize_with_offsets(text)]
