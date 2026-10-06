"""Correct typical OCR errors in ingredient text with a food-domain dictionary (SymSpell).

OCR of small print makes two kinds of mistakes:
  * character slips   "slarch" -> "starch", "maltodexirin" -> "maltodextrin", "regu1ator" -> "regulator"
  * merged words      "acidityregulators" -> "acidity regulators", "palmolein" stays (a real word)

SymSpell (Wolf Garbe's symmetric-delete algorithm) finds dictionary words within a small edit distance
very fast, and `word_segmentation` also splits merged words. Its dictionary is built from the words of our
22,000 real ingredient lists (with their frequencies) plus every dictionary term of the project, so it
"knows" food vocabulary such as maltodextrin, palmolein, carrageenan or jaggery.

Safety rules, so the correction cannot damage text:
  * a word that is known (food dictionary or 82k general English words) is never changed
  * words are corrected one at a time; commas, brackets, colons, numbers and codes
    (E330, INS 471, 20%) are kept exactly
  * corrections can only produce words of the food dictionary, within 1-2 edits
  * the correction is applied to OCR output only, never to text the user typed
"""
import re
from functools import lru_cache
from pathlib import Path

from src.utils.config import project_path

DICTIONARY_PATH = project_path("data/processed/ocr_dictionary.txt")
WORD_RE = re.compile(r"[A-Za-z]+")
LOOKALIKE_DIGITS = str.maketrans({"0": "o", "1": "l", "5": "s", "8": "b", "3": "e", "4": "a"})


@lru_cache(maxsize=1)
def get_symspell():
    """SymSpell loaded with the FOOD dictionary only: corrections can only produce food vocabulary."""
    from symspellpy import SymSpell
    sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    if not sym.load_dictionary(str(DICTIONARY_PATH), term_index=0, count_index=1, separator="\t"):
        raise FileNotFoundError(f"{DICTIONARY_PATH} missing - run python -m src.ocr.build_ocr_dictionary")
    return sym


@lru_cache(maxsize=1)
def known_words() -> frozenset:
    """Words that are never 'corrected': the food dictionary + 82k general English words (shipped with
    symspellpy). A real English word such as 'chili' or 'there' is therefore left alone."""
    import symspellpy
    words = set(get_symspell().words)
    english = Path(symspellpy.__file__).parent / "frequency_dictionary_en_82_765.txt"
    if english.exists():
        words.update(line.split(" ", 1)[0] for line in english.read_text(encoding="utf-8").splitlines())
    return frozenset(words)


def fix_digits_inside_words(text: str) -> str:
    """'dextr0se' -> 'dextrose', 'regu1ator' -> 'regulator'. Only for words that are mostly letters and
    contain at most two digits, so codes like E330, 150d, 5'-guanylate or 25g are never touched."""
    def repair(m):
        word = m.group(0)
        letters = sum(ch.isalpha() for ch in word)
        digits = sum(ch.isdigit() for ch in word)
        if 1 <= digits <= 2 and letters >= 3 and letters >= 2 * digits and word[0].isalpha() \
                and word[-1].isalpha() and not re.match(r"^(?:E|INS)\d", word, re.IGNORECASE):
            return word.translate(LOOKALIKE_DIGITS)
        return word
    return re.sub(r"\b[A-Za-z0-9]*[A-Za-z][A-Za-z0-9]*\b", repair, text)


def _match_case(original: str, corrected: str) -> str:
    if original.isupper():
        return corrected.upper()
    if original[:1].isupper():
        return corrected[:1].upper() + corrected[1:]
    return corrected


def correct_word(word: str) -> str:
    """Correct ONE unknown word: a close food word (edit distance 1 for short words, 2 for long ones),
    or, for long merged words, a split into known food words ('acidityregulator')."""
    lower = word.lower()
    if len(word) <= 3 or lower in known_words():
        return word
    from symspellpy import Verbosity
    sym = get_symspell()
    # short words and ALL-CAPS words (often brand names) only get one edit
    distance = 1 if len(word) <= 6 or word.isupper() else 2
    hits = sym.lookup(lower, Verbosity.TOP, max_edit_distance=distance)
    if hits:
        return _match_case(word, hits[0].term)
    if len(word) >= 9:
        seg = sym.word_segmentation(lower, max_edit_distance=1)
        parts = seg.corrected_string.split()
        if 2 <= len(parts) <= 4 and all(len(p) >= 4 and p in sym.words for p in parts):
            return _match_case(word, " ".join(parts))
    return word


# "rn" read as "m" is the most frequent OCR confusion; "com" is a real word, so fix it only before
# the words that follow "corn" on food labels
CORN_RE = re.compile(r"\b([Cc])om(?=\s*(?:flour|starch|syrup|oil|meal|flakes|grits|solids)\b)", re.IGNORECASE)


# real English words that OCR produces instead of a food word, fixed only in their food context
CONTEXT_FIXES = [
    (re.compile(r"\bchill(?=\s+(?:extract|powder|flakes|sauce|paste|pepper|oleoresin)\b)", re.IGNORECASE), "chilli"),
    (re.compile(r"\bsalf\b", re.IGNORECASE), "salt"),
    (re.compile(r"\b[l1I|]od[il1][sz]ed\b", re.IGNORECASE), "iodised"),        # 'lodised' (I read as l)
    (re.compile(r"(?<=iodised )(?:[s5]?a[l1I]t|sat|sa[l1I]f)\b", re.IGNORECASE), "salt"),   # 'iodised alt'
    (re.compile(r"\b(?:ac[il1]d[il1]ty|acid[il1]t|acdity|cit)\s+regu[l1]ator", re.IGNORECASE), "acidity regulator"),
]


def correct_ocr_text(text: str) -> str:
    """Correct an OCR text word by word; punctuation, numbers, codes and line breaks are kept exactly."""
    text = fix_digits_inside_words(text)
    text = CORN_RE.sub(lambda m: m.group(1) + ("ORN" if m.group(0).isupper() else "orn"), text)
    for pattern, replacement in CONTEXT_FIXES:
        text = pattern.sub(lambda m: _match_case(m.group(0), replacement), text)
    return WORD_RE.sub(lambda m: correct_word(m.group(0)), text)
