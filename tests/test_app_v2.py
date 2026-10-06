"""Tests for the v2 app components: OCR spelling correction, hybrid NER, nutrition parsing,
daily-intake guidance and allergen detection.   Run:  pytest tests/test_app_v2.py -q"""
import pytest

from src.app.allergens import find_allergens
from src.app.intake import PROFILES, additive_guidance, assess_portion, headline, special_notes
from src.app.nutrition import parse_nutrition
from src.ner.hybrid import HybridNER, fuzzy_label
from src.ocr.text_correction import correct_ocr_text, fix_digits_inside_words


# ------------------------------------------------------------------ OCR correction
def test_correction_fixes_typical_ocr_slips():
    out = correct_ocr_text("flavcurenhancers, maltodexrin, slarch, dextr0se, Emu1sifier (INS 322), regu1ator (330)")
    assert out == "flavour enhancers, maltodextrin, starch, dextrose, Emulsifier (INS 322), regulator (330)"


def test_correction_keeps_real_words_codes_and_brands():
    text = "Protein 6.1 g PRINGLES, chili extract, disodium 5'-guanylate, E150d, 25g, their products thereof"
    assert correct_ocr_text(text) == text


def test_correction_context_fixes():
    assert correct_ocr_text("com flour, chill extract") == "corn flour, chilli extract"
    assert correct_ocr_text("come back and chill out") == "come back and chill out"


def test_digits_inside_codes_untouched():
    assert fix_digits_inside_words("E330 150d INS471 5% 503ii") == "E330 150d INS471 5% 503ii"


# ------------------------------------------------------------------ hybrid NER
@pytest.fixture(scope="module")
def hybrid():
    return HybridNER(model=None)          # rules + fuzzy only: fast and deterministic


def labels(result):
    return {e["text"]: e["label"] for e in result["entities"]}


def test_hybrid_fuzzy_recovers_misspelt_sugar_and_sweetener(hybrid):
    found = labels(hybrid.predict("Sugar, glucose syrop, sucralos, invert syrop, wheat flour"))
    assert found["glucose syrop"] == "SUGAR"
    assert found["invert syrop"] == "SUGAR"
    assert found["sucralos"] == "SWEETENER"
    assert found["wheat flour"] == "INGREDIENT"


def test_hybrid_context_rules(hybrid):
    found = labels(hybrid.predict("Carbonated water, colour (caramel E150d), edible vegetable oil (palm), "
                                  "emulsifier: lecithins (soya)"))
    assert found["caramel"] == "COLOUR"
    assert found["palm"] == "FAT"
    assert found["lecithins"] == "ADDITIVE"


def test_hybrid_ignores_warnings_and_keeps_apostrophe_names(hybrid):
    found = labels(hybrid.predict("disodium 5'-guanylate, acidity regulator (citric acid) PHENYLKETONURICS: "
                                  "CONTAINS PHENYLALANINE"))
    assert found["disodium 5'-guanylate"] == "ADDITIVE"
    assert found["citric acid"] == "ADDITIVE"
    assert "PHENYLKETONURICS" not in found


def test_fuzzy_does_not_touch_real_words():
    assert fuzzy_label("rolled oat") is None
    assert fuzzy_label("salt") is None


# ------------------------------------------------------------------ nutrition
EU_TABLE = """NUTRITION INFORMATION Typical values per 100g
Energy 2252 kJ / 539 kcal
Fat 30.9 g
of which saturates 10.6 g
Carbohydrate 57.5 g
of which sugars 56.3 g
Protein 6.3 g
Salt 0.107 g"""


def test_parse_eu_table():
    per100 = parse_nutrition(EU_TABLE)["per_100g"]
    assert per100 == {"energy_kcal": 539.0, "fat_g": 30.9, "saturated_fat_g": 10.6, "carbohydrate_g": 57.5,
                      "sugars_g": 56.3, "protein_g": 6.3, "salt_g": 0.11}


def test_parse_kj_only_energy_and_sodium():
    parsed = parse_nutrition("per 100 g\nEnergy 2214 kcal\nSodium 400 mg")["per_100g"]
    assert parsed["energy_kcal"] == pytest.approx(529.2, abs=0.1)     # >900 kcal is impossible -> kJ
    assert parsed["salt_g"] == 1.0                                      # 0.4 g sodium x 2.5


def test_parse_us_label_per_serving_is_converted():
    parsed = parse_nutrition("Serving size 1 cup (240mL)\nCalories 140\nTotal Sugars 39g\nProtein 0g")
    assert parsed["serving_g"] == 240
    assert parsed["per_100g"]["sugars_g"] == pytest.approx(16.25)


# ------------------------------------------------------------------ intake guidance
def test_portion_shares_and_limits():
    rows = {r["key"]: r for r in assess_portion({"sugars_g": 50.0, "salt_g": 1.0}, 50, PROFILES["Adult (2000 kcal)"])}
    assert rows["sugars_g"]["in_portion"] == 25.0
    assert rows["sugars_g"]["share"] == pytest.approx(0.5)              # WHO: 50 g/day at 2000 kcal
    assert rows["sugars_g"]["max_product_g"] == 100
    assert rows["sugars_g"]["traffic_light"] == "high"
    assert rows["salt_g"]["daily_value"] == 5.0
    text, level = headline(list(rows.values()), 50)
    assert level == "high"


def test_child_profile_has_lower_limits():
    adult = {r["key"]: r for r in assess_portion({"sugars_g": 10.0}, 100, PROFILES["Adult (2000 kcal)"])}
    child = {r["key"]: r for r in assess_portion({"sugars_g": 10.0}, 100, PROFILES["Child 4-6 (1400 kcal)"])}
    assert child["sugars_g"]["share"] > adult["sugars_g"]["share"]


def test_additive_adi_scaled_by_body_weight():
    rows = additive_guidance([{"ins_number": "951", "canonical_name": "Aspartame", "text": "aspartame"},
                              {"ins_number": "330", "canonical_name": "Citric acid", "text": "INS 330"}], 60)
    by_name = {r["additive"]: r for r in rows}
    assert "2,400 mg/day" in by_name["Aspartame (INS 951)"]["daily_limit"]
    assert "not specified" in by_name["Citric acid (INS 330)"]["daily_limit"]
    assert any(n["substance"].startswith("Caffeine") for n in special_notes("water, sugar, caffeine"))


# ------------------------------------------------------------------ allergens
def test_allergens_contains_and_may_contain():
    found = find_allergens("Wheat flour, sugar, coconut, cocoa butter, milk solids. May contain traces of nuts and sesame.")
    assert set(found["contains"]) == {"Cereals containing gluten", "Milk"}
    assert set(found["may_contain"]) == {"Tree nuts", "Sesame"}


# ------------------------------------------------------------------ end-to-end photo (OCR) incl. orientation
@pytest.mark.parametrize("rotation", [None, "cw", "180"])
def test_photo_end_to_end_any_orientation(rotation):
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("rapidocr")
    from src.app.scanner import IngredientScanner
    from src.utils.config import project_path
    image = cv2.imread(str(project_path("data/images/packets/8901499009289.jpg")))
    if rotation:
        image = cv2.rotate(image, {"cw": cv2.ROTATE_90_CLOCKWISE, "180": cv2.ROTATE_180}[rotation])
    scanner = IngredientScanner("dictionary")
    scanner.ner = HybridNER(model=None)
    result = scanner.scan_image(image)
    sugars = {s.lower() for s in result["summary"]["groups"]["SUGAR"]}
    assert {"sugar", "dextrose", "honey"} <= sugars
    assert any("150d" in c for c in result["summary"]["groups"]["COLOUR"])
