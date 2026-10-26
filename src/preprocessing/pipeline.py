"""The single entry point for preprocessing, used by every part of the project.

    from src.preprocessing.pipeline import process_ingredient_text
    result = process_ingredient_text("SUGAR, Glucose Syrup, Acidity Regulator (INS 330)")

It works on ANY string: Open Food Facts text, OCR output from Person 3's EasyOCR pipeline, or
text typed by a user. It never depends on Open Food Facts columns.
"""
from src.preprocessing.additive_codes import find_additive_codes
from src.preprocessing.normalize import non_latin_letter_ratio, normalize_text
from src.preprocessing.segment import extract_percentages, segment_ingredients
from src.preprocessing.tokenize import tokenize_with_offsets


def process_ingredient_text(text) -> dict:
    """Normalise, tokenise and analyse one ingredient list.

    Returns a dict:
      raw_text        the input, unchanged
      text            normalised text (all offsets below refer to THIS string)
      tokens          word-level tokens            ["SUGAR", ",", "Glucose", ...]
      token_offsets   [[start, end], ...] per token
      segments        ingredient segments with bracket depth (compound ingredients)
      additive_codes  INS/E codes with canonical number, e.g. {"text": "INS 330", "number": "330"}
      percentages     percentages with value, e.g. {"text": "20.7%", "value": 20.7}
      non_latin_ratio share of non-Latin letters (e.g. Arabic part of a bilingual label)
    """
    raw = "" if text is None else str(text)
    normalized = normalize_text(raw)
    tokens = tokenize_with_offsets(normalized)
    codes = find_additive_codes(normalized)
    segments = segment_ingredients(normalized, protected_spans=[(c["start"], c["end"]) for c in codes])
    return {
        "raw_text": raw,
        "text": normalized,
        "tokens": [t for t, _, _ in tokens],
        "token_offsets": [[s, e] for _, s, e in tokens],
        "segments": segments,
        "additive_codes": codes,
        "percentages": extract_percentages(normalized),
        "non_latin_ratio": round(non_latin_letter_ratio(normalized), 3),
    }
