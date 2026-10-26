"""Find INS / E-number additive codes in text and normalise them.

Labels write the same additive in many ways:
    "E330"  "E 330"  "e-330"  "INS 330"  "INS-330"  "INS No. 330"  "(330)"
    sub-types: "E150d"  "503(ii)"  "E500ii"  "INS 341 (i)"  "160a(ii)"
All of these become one canonical number in Open Food Facts tag form: "330", "150d", "503ii".
(INS and E numbers are the same numbering system: INS 330 = E330 = citric acid.)

Three patterns:
  1. E-number  "E" + number                    -> always a code (except "vitamin E 400 IU")
  2. INS       "INS" + number                  -> always a code
  3. bare      a number on its own, e.g. "Emulsifier (471)" or "Thickeners (508 & 412)".
     A bare number is ambiguous ("100 g", "2015"), so it only counts as a code when
       * it is a known additive number (in the OFF reference table), and
       * it is not followed by a unit or "%", and
       * a functional class word ("emulsifier", "colour" ...) or another code appears shortly
         before it in the same sentence.
"""
import re
from functools import lru_cache

from src.labeling.lexicon import additive_reference, function_class_terms

ROMAN = r"(?:viii|vii|iii|ii|iv|vi|ix|v|i|x)"
# number + optional letter + optional roman sub-number, e.g. 330 | 150d | 503(ii) | 500ii | 160a(ii)
NUMBER = rf"(?P<num>\d{{3,4}})(?P<letter>[a-f])?(?:\s?\(\s?(?P<roman1>{ROMAN})\s?\)|(?P<roman2>{ROMAN}))?"
END = r"(?![\w%]|[.,]\d)"  # the code must not continue into a word, a percentage or a decimal ("955.9%")

E_CODE_RE = re.compile(rf"\bE\s?-?\s?{NUMBER}{END}", re.IGNORECASE)
INS_CODE_RE = re.compile(rf"\bINS\s?(?:No\.?|Number)?\s?[:.\-]?\s?{NUMBER}{END}", re.IGNORECASE)
BARE_CODE_RE = re.compile(rf"(?<![\w.,/-]){NUMBER}{END}", re.IGNORECASE)
UNIT_AFTER_RE = re.compile(r"^\s?(?:%|g\b|mg\b|kg\b|mcg\b|µg\b|ml\b|l\b|kcal\b|kj\b|iu\b|cal\b|ppm\b)",
                           re.IGNORECASE)
CONTEXT_WINDOW = 80  # characters looked at before a bare number


def canonical_number(match) -> str:
    """Regex match -> '330', '150d', '503ii'."""
    roman = match.group("roman1") or match.group("roman2") or ""
    return (match.group("num") + (match.group("letter") or "") + roman).lower()


@lru_cache(maxsize=1)
def _context_re():
    """Regex that finds a functional class word or an E/INS code (evidence that numbers nearby are codes)."""
    terms = sorted(function_class_terms(), key=len, reverse=True)
    words = "|".join(re.escape(t) for t in terms)
    return re.compile(rf"\b(?:{words})\b|\bINS\b|\bE\s?\d{{3}}", re.IGNORECASE)


def _is_known_number(number: str) -> bool:
    reference = additive_reference()
    base = re.match(r"\d+[a-f]?", number).group(0)
    return number in reference or base in reference or re.match(r"\d+", number).group(0) in reference


def _bare_number_is_code(text: str, match) -> bool:
    if not _is_known_number(canonical_number(match)):
        return False
    if UNIT_AFTER_RE.match(text[match.end():]):
        return False
    window = text[max(0, match.start() - CONTEXT_WINDOW):match.start()]
    window = re.split(r"\.\s", window)[-1]          # stay inside the current sentence
    return bool(_context_re().search(window))


def find_additive_codes(text: str) -> list:
    """Return codes found in `text`, ordered by position, without overlaps.

    Each item: {"start", "end", "text", "number", "style"} where style is "E", "INS" or "bare".
    Character offsets refer to `text` exactly.
    """
    found = []

    def overlaps(start, end):
        return any(start < f["end"] and f["start"] < end for f in found)

    for style, regex in (("INS", INS_CODE_RE), ("E", E_CODE_RE)):
        for m in regex.finditer(text):
            if style == "E" and re.search(r"vitamin\s*$", text[:m.start()], re.IGNORECASE):
                continue  # "vitamin E 400 IU"
            if not overlaps(m.start(), m.end()):
                found.append({"start": m.start(), "end": m.end(), "text": m.group(0),
                              "number": canonical_number(m), "style": style})

    for m in BARE_CODE_RE.finditer(text):
        if not overlaps(m.start(), m.end()) and _bare_number_is_code(text, m):
            found.append({"start": m.start(), "end": m.end(), "text": m.group(0),
                          "number": canonical_number(m), "style": "bare"})
    return sorted(found, key=lambda f: f["start"])


def normalize_code(code_text: str):
    """'INS 503(ii)' / 'E503ii' / '503 (ii)' -> '503ii'; returns None if it is not a code."""
    for regex in (INS_CODE_RE, E_CODE_RE, BARE_CODE_RE):
        m = regex.fullmatch(code_text.strip())
        if m:
            return canonical_number(m)
    return None
