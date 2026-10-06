"""Targeted correction of OCR errors in additive codes.

OCR often confuses digits with similar letters: "(I5Od)" instead of "(150d)", "INS 33O" instead of
"INS 330". We correct ONLY short code-like tokens in places where an additive code is expected
(inside brackets, or right after "INS" / "E"), and only if the token already contains a real digit.
Ordinary words are never changed, so the correction cannot invent ingredients.

    fix_additive_codes("Colour (I5Od), INS 33O, Milk (73%)") -> "Colour (150d), INS 330, Milk (73%)"
"""
import re

LOOKALIKE = str.maketrans({"I": "1", "l": "1", "|": "1", "i": "1", "O": "0", "o": "0", "Q": "0", "D": "0",
                           "S": "5", "B": "8", "Z": "2"})
# 3-4 characters that are digits or look-alike letters, optionally followed by a sub-type letter a-f
TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])([0-9IlOoQDSBZi|]{3,4})([a-f]?)(?![A-Za-z0-9])")
CONTEXT_RE = re.compile(r"\(([^()]{1,60})\)|\[([^\[\]]{1,60})\]|\b(?:INS|E)\s?-?\s?([0-9IlOoSB|]{3,4}[a-f]?)\b")


def _fix_tokens(segment: str) -> str:
    def repair(m):
        core, letter = m.group(1), m.group(2)
        if not any(ch.isdigit() for ch in core):          # e.g. "Oil" is never touched
            return m.group(0)
        return core.translate(LOOKALIKE) + letter
    return TOKEN_RE.sub(repair, segment)


def fix_additive_codes(text: str) -> str:
    def repair_context(m):
        if m.group(3):                                    # "INS 33O", "E2O2": fix the code after the prefix
            code = m.group(3)
            if not any(ch.isdigit() for ch in code):
                return m.group(0)
            fixed = code[:-1].translate(LOOKALIKE) + code[-1] if code[-1] in "abcdef" else code.translate(LOOKALIKE)
            return m.group(0)[: m.start(3) - m.start(0)] + fixed
        return _fix_tokens(m.group(0))
    return CONTEXT_RE.sub(repair_context, text)
