"""The characters the recogniser can output, and text <-> label-id conversion for CTC.

Id 0 is the CTC "blank" (no character at this time step); real characters start at 1.
"""
import string
import unicodedata

CHARS = " " + string.digits + string.ascii_letters + ",.;:()[]%-&/'+*!?#\""
BLANK = 0
CHAR2ID = {ch: i + 1 for i, ch in enumerate(CHARS)}
ID2CHAR = {i: ch for ch, i in CHAR2ID.items()}
NUM_CLASSES = len(CHARS) + 1          # + blank

# characters that are not in CHARS but have a close ASCII equivalent
_REPLACE = {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "•": ",", "·": ".", "×": "x"}


def clean_text(text: str) -> str:
    """Map a reference text into the character set: accents removed ('é' -> 'e'), unknown characters dropped,
    whitespace collapsed."""
    text = "".join(_REPLACE.get(ch, ch) for ch in text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if ch in CHAR2ID or ch.isspace())
    return " ".join(text.split())


def encode(text: str) -> list:
    return [CHAR2ID[ch] for ch in text if ch in CHAR2ID]


def decode_ids(ids) -> str:
    return "".join(ID2CHAR.get(int(i), "") for i in ids)
