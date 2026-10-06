"""Tests for src/preprocessing (normalisation, tokenisation, codes, segmentation, public API)."""
from src.preprocessing.additive_codes import find_additive_codes, normalize_code
from src.preprocessing.normalize import normalize_text
from src.preprocessing.pipeline import process_ingredient_text
from src.preprocessing.segment import extract_percentages, segment_ingredients
from src.preprocessing.tokenize import tokenize, tokenize_with_offsets


def codes(text):
    return [(c["text"], c["number"]) for c in find_additive_codes(text)]


def test_normalize_keeps_information():
    assert normalize_text("Acidity Regulator (INS 330)") == "Acidity Regulator (INS 330)"   # no lowercasing
    assert normalize_text("E330, 20.5%") == "E330, 20.5%"                                     # numbers kept


def test_normalize_cleans_noise():
    assert normalize_text("_Wheat_ Flour") == "Wheat Flour"                  # OFF allergen markup
    assert normalize_text("Milk &amp; Soy") == "Milk & Soy"                  # HTML entity
    assert normalize_text("ＩＮＳ ３３０") == "INS 330"                        # full-width characters (NFKC)
    assert normalize_text("emulsi-\nfier ,  salt") == "emulsifier , salt"    # OCR line-break hyphen, spaces
    assert normalize_text("sugar • salt") == "sugar , salt"                  # bullet -> separator
    assert normalize_text("sugar,, salt") == "sugar, salt"
    assert normalize_text(None) == "" and normalize_text(float("nan")) == ""


def test_normalize_is_idempotent():
    text = "  SUGAR (20%),, _Milk_ &amp; INS-330 • salt\n"
    once = normalize_text(text)
    assert normalize_text(once) == once


def test_tokenize_with_offsets():
    text = "INS 330, 503(ii), sugar (20.7%), baker's yeast"
    tokens = tokenize_with_offsets(text)
    assert [t for t, _, _ in tokens] == ["INS", "330", ",", "503", "(", "ii", ")", ",", "sugar", "(", "20.7%",
                                         ")", ",", "baker's", "yeast"]
    assert all(text[s:e] == t for t, s, e in tokens)
    assert tokenize("E150d") == ["E150d"]


def test_additive_code_variants():
    assert codes("E330, E 330, e-330, INS 330, INS-330, INS No. 330") == [
        ("E330", "330"), ("E 330", "330"), ("e-330", "330"), ("INS 330", "330"), ("INS-330", "330"),
        ("INS No. 330", "330")]
    assert codes("Colour (E150d), raising agent 503(ii), INS 341 (i)") == [
        ("E150d", "150d"), ("503(ii)", "503ii"), ("INS 341 (i)", "341i")]


def test_bare_numbers_need_context():
    assert codes("Emulsifier (471), Thickeners (508 & 412)") == [("471", "471"), ("508", "508"), ("412", "412")]
    assert codes("Sugar 100 g, 330 mg sodium, best before 2025") == []       # quantities / years are not codes
    assert codes("Vitamin E 400 IU") == []
    assert codes("SWEETENERS (955.9% x") == []                               # decimal, not a code


def test_normalize_code():
    assert normalize_code("INS 503(ii)") == normalize_code("E503ii") == normalize_code("503 (ii)") == "503ii"
    assert normalize_code("hello") is None


def test_segmentation_and_percentages():
    text = "Chocolate (sugar, cocoa butter 5,5%), Salt; Emulsifier: soy lecithin. Water"
    segs = [(s["text"], s["depth"]) for s in segment_ingredients(text)]
    assert segs == [("Chocolate", 0), ("sugar", 1), ("cocoa butter 5,5%", 1), ("Salt", 0), ("Emulsifier", 0),
                    ("soy lecithin", 0), ("Water", 0)]
    assert [p["value"] for p in extract_percentages("Sugar (20.7%), Cocoa 5 %, 5,5%")] == [20.7, 5.0, 5.5]


def test_process_ingredient_text_contract():
    """The public function used by Person 3 (OCR): any string in, consistent fields out."""
    result = process_ingredient_text("SUGAR, Acidity Regulator (INS 330)\nemulsi-\nfier")
    assert result["raw_text"].startswith("SUGAR")
    assert result["text"] == "SUGAR, Acidity Regulator (INS 330) emulsifier"
    assert len(result["tokens"]) == len(result["token_offsets"])
    assert result["additive_codes"][0]["number"] == "330"
    assert [s["text"] for s in result["segments"]] == ["SUGAR", "Acidity Regulator", "INS 330", "emulsifier"]
    empty = process_ingredient_text("")
    assert empty["tokens"] == [] and empty["segments"] == []
