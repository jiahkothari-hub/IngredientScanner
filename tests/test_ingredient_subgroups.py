"""Tests for the sub-groups of generic ingredients (src/kb/ingredient_subgroups.py), how the scanner and the
assistant use them, and monk fruit as a sweetener.   Run:  pytest tests/test_ingredient_subgroups.py -q"""
import pytest

from src.kb.ingredient_subgroups import INFO, ORDER, rule_subgroup, subgroup_of


@pytest.mark.parametrize("name,group", [
    ("Chicory Root Fiber", "fibre"), ("Probiotic Bacillus Coagulans", "culture"),
    ("Saccharomyces cerevisiae", "culture"), ("Bromelain & Papain", "enzyme"),
    ("caffeine from coffee powder", "stimulant"), ("Cocoa Powder", "cocoa_coffee"),
    ("Refined Wheat Flour (Maida)", "grain"), ("Gram Meal", "legume"), ("Milk Solids", "dairy"),
    ("Iodised Salt", "salt"), ("Tomato Paste", "fruit_veg"), ("kasuri methi", "spice_herb"),
    ("Elderberry Extract", "fruit_veg")])
def test_rules(name, group):
    assert rule_subgroup(name) == group


def test_rule_order_resolves_overlaps():
    assert rule_subgroup("Whey protein concentrate") == "protein"          # protein before dairy
    assert rule_subgroup("Fermented Yeast Protein Isolate") == "protein"   # protein before yeast
    assert rule_subgroup("Hydrolysed Groundnut Protein") == "seasoning"    # a flavour base, not a protein source


def test_unknown_names_fall_back_to_other_or_embeddings():
    group, method = subgroup_of("zzqqxx")
    assert (group, method) == ("other", "")
    assert set(ORDER) == set(INFO)


def test_monk_fruit_is_a_sweetener():
    from src.labeling.weak_labeler import label_text
    labels = {e["text"]: e["label"] for e in label_text("Monk Fruit Powder, cocoa")["entities"]}
    assert labels["Monk Fruit Powder"] == "SWEETENER"


@pytest.fixture(scope="module")
def protein_powder():
    from src.app.scanner import IngredientScanner
    return IngredientScanner("dictionary").scan_text(
        "Ingredients: Fermented Yeast Protein Isolate, Chicory Root Fiber, Cocoa Powder, Monk Fruit Powder, "
        "Natural Digestive Enzymes, Probiotic Bacillus Coagulans, caffeine from coffee powder")


def test_scan_summary_has_subgroups(protein_powder):
    sub = protein_powder["summary"]["subgroups"]
    assert [n["name"] for n in sub["culture"]] == ["Probiotic Bacillus Coagulans"]
    assert {"protein", "fibre", "enzyme", "stimulant", "cocoa_coffee"} <= set(sub)
    assert protein_powder["summary"]["groups"]["SWEETENER"] == ["Monk Fruit Powder"]


def test_assistant_answers_subgroup_questions(protein_powder):
    from src.assistant.assistant import IngredientAssistant
    bot = IngredientAssistant(protein_powder)
    assert "Bacillus Coagulans" in bot.answer("which ingredients are probiotics?")["text"]
    assert "Enzymes" in bot.answer("does it contain any enzymes?")["text"]
    assert "Monk Fruit Powder" in bot.answer("what sweeteners does it use?")["text"]
    caffeine = bot.answer("what is caffeine?")["text"]
    assert "Caffeine & stimulants" in caffeine and "400 mg" in caffeine
    assert "Which ingredients are probiotics?" in bot.suggestions()


def test_subgroup_card_renders():
    from src.app.ui_components import subgroup_card
    html = subgroup_card(["a", "b"], {"fibre": [{"name": "Inulin", "estimated": False}],
                                      "fruit_veg": [{"name": "Sultanas", "estimated": True}]})
    assert "Fibres" in html and "Sultanas ~" in html and "estimate" in html
