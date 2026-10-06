"""Text normalisation for ingredient lists (Open Food Facts text AND OCR output).

Principle: change as little as possible. We only remove things that carry no meaning for NER
(markup, invisible characters, inconsistent spacing) and unify characters that have several
Unicode forms. We do NOT lowercase, remove numbers, remove punctuation or rewrite additive codes:
"INS 330", "E330" and "(330)" must all survive, because they are entities.

The original text is never overwritten - callers keep it in a separate field.
"""
import html
import re
import unicodedata

# Characters with several Unicode spellings -> one ASCII spelling
CHAR_MAP = {
    "‘": "'", "’": "'", "‚": "'", "′": "'",       # curly / prime apostrophes
    "“": '"', "”": '"', "„": '"',                       # curly double quotes
    "‐": "-", "‑": "-", "‒": "-", "–": "-",        # hyphens and dashes
    "—": "-", "―": "-", "−": "-",
    "•": ",", "·": ",", "●": ",", "▪": ",",        # bullets used as separators
    "∙": ",", "،": ",",                                      # bullet operator, Arabic comma
    "®": "", "™": "", "©": "",                          # (R) (TM) (C) carry no meaning
}
_CHAR_TABLE = str.maketrans(CHAR_MAP)


def normalize_unicode(text: str) -> str:
    """Decode HTML entities, apply NFKC and unify quote/dash/bullet characters.

    NFKC turns compatibility characters into their plain form, e.g. full-width
    letters "ＩＮＳ" -> "INS" and the ligature "ﬂ" -> "fl" (common in PDF/OCR text).
    """
    text = html.unescape(html.unescape(text))   # twice: some texts are double-escaped (&amp;quot;)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_CHAR_TABLE)
    # Remove invisible control/format characters (zero-width space etc.), keep newlines/tabs for now
    return "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch) not in ("Cc", "Cf"))


def remove_markup(text: str) -> str:
    """Open Food Facts marks allergens with underscores: '_Wheat_ Flour' -> 'Wheat Flour'."""
    return text.replace("_", "")


def normalize_whitespace(text: str) -> str:
    """Join words split across lines by OCR ('emulsi-\\nfier'), then collapse all whitespace."""
    text = re.sub(r"(?<=[^\W\d_])-[ \t]*\n[ \t]*(?=[a-z])", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fix_repeated_separators(text: str) -> str:
    """',,' or ', ,' -> ','  (common after OCR or bullet replacement)."""
    text = re.sub(r",(\s*,)+", ",", text)
    return re.sub(r"^[,;\s]+|[,;\s]+$", "", text)


def normalize_text(text) -> str:
    """Full normalisation used everywhere in the project. Safe to call on any string or None."""
    if text is None or (isinstance(text, float) and text != text):  # None or NaN
        return ""
    text = normalize_unicode(str(text))
    text = remove_markup(text)
    text = normalize_whitespace(text)
    return fix_repeated_separators(text)


def non_latin_letter_ratio(text: str) -> float:
    """Share of letters that are not Latin script (e.g. Arabic part of a bilingual label)."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    non_latin = sum(1 for ch in letters if "LATIN" not in unicodedata.name(ch, ""))
    return non_latin / len(letters)
