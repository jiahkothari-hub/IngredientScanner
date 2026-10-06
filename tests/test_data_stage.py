"""Tests for Stage 1 (data acquisition and filtering). Run with:  python -m pytest -q"""
from pathlib import Path

import pandas as pd

from src.data.download_taxonomy import build_additives_table, parse_taxonomy
from src.data.eda import code_style, crude_segments
from src.data.filter_dataset import assign_country_group, dedup_key, filter_candidates, is_placeholder
from src.data.language import detect_language

GROUPS = {"india": ["India"], "uk_ireland": ["United Kingdom", "Ireland"]}


def test_detect_language():
    assert detect_language("Sugar, wheat flour, palm oil, salt")[0] == "en"
    assert detect_language("Sucre, farine de blé, huile de palme, sel")[0] == "fr"
    assert detect_language("Zucker, Weizenmehl, Salz")[0] == "de"
    assert detect_language("E330, E211") == ("unknown", 0.0)
    # bilingual label: half English, half French
    assert detect_language("Sugar, salt / sucre, sel")[1] == 0.5


def test_assign_country_group_uses_config_order():
    assert assign_country_group("United Kingdom,India", GROUPS) == "india"
    assert assign_country_group("Ireland", GROUPS) == "uk_ireland"
    assert assign_country_group("France", GROUPS) is None
    assert assign_country_group(float("nan"), GROUPS) is None


def test_dedup_key_ignores_case_and_spacing():
    assert dedup_key("Sugar ,  SALT") == dedup_key("sugar, salt")


def test_filter_candidates_keeps_raw_text_untouched():
    raw = "  Sugar, Palm Oil,  INS 330  "
    df = pd.DataFrame({
        "code": ["1", "2", "2", "3", "4", "5", "6", "7"],
        "ingredients_text": [raw, "sugar, salt", "sugar, salt", "Sucre, sel, farine", None, "undefined",
                             "Bing cherries.", "Sugar, salt, water. Ingrédients: sucre, sel, eau"],
        "country_group": ["india"] * 8,
    })
    config = {"random_seed": 0, "filter": {"min_chars": 3, "max_chars": 1500, "min_english_score": 0.9,
                                           "group_quota": {"india": None}}}
    out, log = filter_candidates(df, config)
    # removed: duplicate barcode, French, empty, "undefined", bilingual EN/FR
    assert sorted(out["code"]) == ["1", "2", "6"]
    assert out.loc[out["code"] == "1", "ingredients_text"].item() == raw  # original text preserved
    assert out.loc[out["code"] == "6", "english_rule"].item() == "assumed_ascii_no_markers"
    assert log["funnel"][-1]["products"] == 3


def test_is_placeholder():
    assert is_placeholder("undefined") and is_placeholder("  N/A ") and is_placeholder(None)
    assert not is_placeholder("sugar")


def test_code_style_and_segments():
    assert code_style("Acidity regulator (INS 330)") == "INS number (INS 330)"
    assert code_style("Colour: E150d") == "E-number (E330)"
    assert code_style("Emulsifier (322), sugar") == "bare number (330)"
    assert code_style("Sugar, salt") == "no code"
    assert crude_segments("Sugar (30%), Cocoa Butter; Milk") == ["sugar", "cocoa butter", "milk"]


def test_parse_taxonomy(tmp_path: Path):
    text = (
        "# comment\n\n"
        "en: E330, Citric acid\nxx: E330\nfr: E330, Acide citrique\n"
        "additives_classes:en: en:acid, en:antioxidant\ne_number:en: 330\n\n"
        "en: E129, Allura red, FD&C Red 40\ne_number:en: 129\nadditives_classes:en: en:colour\n\n"
        "en: E322, Lecithins\ne_number:en: 322\nadditives_classes:en: en:emulsifier\n\n"
        "en: E322(i), Lecithin\n\n"            # sub-type: no e_number property, no classes
        "en: E160d(iii), Lycopene\\, Blakeslea trispora\n"
    )
    path = tmp_path / "additives.txt"
    path.write_text(text, encoding="utf-8")
    table = build_additives_table(parse_taxonomy(path))
    citric = table[table["e_code"] == "E330"].iloc[0]
    assert citric["name"] == "Citric acid"
    assert citric["function_classes"] == "acid|antioxidant"
    assert "FD&C Red 40" in table[table["e_code"] == "E129"].iloc[0]["synonyms"]
    sub = table[table["e_code"] == "E322i"].iloc[0]                     # E322(i) -> tag form 322i
    assert sub["function_classes"] == "emulsifier" and sub["classes_source"] == "inherited_from_parent"
    assert table[table["e_code"] == "E160diii"].iloc[0]["name"] == "Lycopene, Blakeslea trispora"
