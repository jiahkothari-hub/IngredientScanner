"""Split an ingredient list into ingredient segments, and extract percentages.

    "Chocolate (sugar, cocoa butter), Salt; Emulsifier: soy lecithin"
      -> "Chocolate" (depth 0)  "sugar" (1)  "cocoa butter" (1)  "Salt" (0)  "Emulsifier" (0)  "soy lecithin" (0)

Separators: , ; : |  and the brackets ( ) [ ] { }, plus a full stop that ends a sentence.
`depth` counts how many brackets surround the segment, so "sugar" above is a SUB-ingredient of
"Chocolate" (a compound ingredient). Additive codes such as "503(ii)" are passed in as protected
spans so their brackets are not treated as separators.
"""
import re

OPEN, CLOSE = "([{", ")]}"
SEPARATORS = ",;:|"
# a full stop after these words is an abbreviation, not the end of a sentence
ABBREVIATIONS = {"no", "nos", "vit", "approx", "st", "mr", "dr", "u.s.p", "usp", "inc", "ltd", "co", "fd", "e.g", "i.e"}
TRIM_CHARS = " *†‡.-'\"!?#&+/"
PERCENT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s?%")


def _is_sentence_end(text: str, i: int) -> bool:
    """Is the '.' at position i the end of a sentence (not '2.5', 'No. 330', 'u.s.p.')?"""
    if i + 1 < len(text) and not text[i + 1].isspace():
        return False                                   # "2.5", "u.s.p" - the dot is inside a token
    previous_word = re.search(r"([\w.]+)$", text[:i])
    if previous_word and previous_word.group(1).lower() in ABBREVIATIONS:
        return False
    if previous_word and len(previous_word.group(1)) == 1 and previous_word.group(1).isalpha():
        return False                                   # single letters: "vitamin B. ..." is rare, initials common
    return True


def segment_ingredients(text: str, protected_spans=()) -> list:
    """Return [{"start", "end", "text", "depth"}, ...] with offsets into `text`."""
    protected = sorted(protected_spans)
    segments, depth, seg_start, i = [], 0, 0, 0

    def close(end, seg_depth):
        raw = text[seg_start:end]
        left = len(raw) - len(raw.lstrip(TRIM_CHARS))
        right = len(raw.rstrip(TRIM_CHARS))
        if right > left:
            segments.append({"start": seg_start + left, "end": seg_start + right,
                             "text": raw[left:right], "depth": seg_depth})

    while i < len(text):
        span = next(((s, e) for s, e in protected if s == i), None)
        if span:                       # jump over an additive code such as "503(ii)"
            i = span[1]
            continue
        ch = text[i]
        if ch in OPEN:
            close(i, depth)
            depth += 1
            seg_start = i + 1
        elif ch in CLOSE:
            close(i, depth)
            depth = max(0, depth - 1)
            seg_start = i + 1
        elif ch == "," and 0 < i < len(text) - 1 and text[i - 1].isdigit() and text[i + 1].isdigit():
            pass                       # decimal comma, "5,5%" (European number format)
        elif ch in SEPARATORS or (ch == "." and _is_sentence_end(text, i)):
            close(i, depth)
            seg_start = i + 1
        i += 1
    close(len(text), depth)
    return segments


def extract_percentages(text: str) -> list:
    """'Sugar (20.7%), Cocoa 5 %' -> [{'value': 20.7, 'start':.., 'end':.., 'text': '20.7%'}, ...]"""
    return [{"value": float(m.group(1).replace(",", ".")), "start": m.start(), "end": m.end(), "text": m.group(0)}
            for m in PERCENT_RE.finditer(text)]
