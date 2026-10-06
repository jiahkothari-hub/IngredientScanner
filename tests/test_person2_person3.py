"""Tests for the NER models (Person 2) and OCR / knowledge base / app (Person 3) components.
Model-dependent tests are skipped when the (large) model files are not present."""
import pytest

from src.app.scanner import IngredientScanner, summarise
from src.kb.entity_linking import get_linker
from src.ner.crf_baseline import sentence_features, word_shape
from src.ner.data import noisy_copy
from src.ocr.ocr_cleanup import fix_additive_codes
from src.ocr.section_extraction import extract_ingredients_section
from src.utils.config import project_path


# ------------------------------------------------------------------ OCR helpers
def test_ocr_code_cleanup_only_touches_codes():
    assert fix_additive_codes("Colour (I5Od), INS 33O, E2O2") == "Colour (150d), INS 330, E202"
    assert fix_additive_codes("Raising agents [5O3(ii), 5OO(ii)]") == "Raising agents [503(ii), 500(ii)]"
    assert fix_additive_codes("Oil (Palm), Eggs, Salt (1.5%), INS Oil") == "Oil (Palm), Eggs, Salt (1.5%), INS Oil"


def test_section_extraction():
    ocr = "BRAND\nTasty Biscuits\nINGREDIENTS: Wheat flour, sugar,\npalm oil, salt.\nNUTRITION INFORMATION per 100g\nEnergy 500 kcal"
    section = extract_ingredients_section(ocr)
    assert section["method"] == "keyword"
    assert section["text"] == "Wheat flour, sugar, palm oil, salt"
    assert extract_ingredients_section("lngredients: sugar, salt")["text"] == "sugar, salt"       # OCR 'l' for 'I'
    no_keyword = extract_ingredients_section("BRAND\nwheat flour, sugar, oil, salt, yeast, malt\nNet wt 200g")
    assert no_keyword["method"] == "comma_density" and no_keyword["text"].startswith("wheat flour")


# ------------------------------------------------------------------ knowledge base / linking
def test_entity_linking_methods():
    linker = get_linker()
    assert linker.link({"text": "INS 330", "label": "INS_CODE"})["canonical_name"] == "Citric acid"
    assert linker.link({"text": "500(ii)", "label": "INS_CODE"})["ins_number"] == "500ii"
    exact = linker.link({"text": "soy lecithin", "label": "ADDITIVE"})
    assert exact["link_method"] == "exact_name" and exact["ins_number"] == "322"
    fuzzy = linker.link({"text": "potasium sorbate", "label": "PRESERVATIVE"})
    assert fuzzy["link_method"] == "fuzzy_name" and fuzzy["canonical_name"] == "Potassium sorbate"
    assert linker.link({"text": "wheat flour", "label": "INGREDIENT"})["link_method"] == "category"


def test_scanner_text_mode_with_rules():
    scanner = IngredientScanner("dictionary")
    result = scanner.scan_text("Sugar, dextrose, Acidity regulator (INS 330), palm oil, preservative (211)")
    summary = result["summary"]
    assert summary["groups"]["SUGAR"] == ["Sugar", "dextrose"]
    assert summary["hidden"]["sugars_under_other_names"] == ["dextrose"]
    assert "INS 330 (Citric acid)" in summary["groups"]["ADDITIVE"]
    assert "211 (Sodium benzoate)" in summary["groups"]["PRESERVATIVE"]
    ins = next(e for e in result["entities"] if e["text"] == "INS 330")
    assert ins["function"] == "acidity regulator" and ins["function_source"] == "label"
    assert "not medical advice" in result["disclaimer"]


def test_summary_skips_function_words():
    items = [{"text": "Emulsifier", "label": "FUNCTION_CLASS", "kb_category": "FUNCTION_CLASS", "canonical_name": ""}]
    assert sum(summarise(items)["counts"].values()) == 0


# ------------------------------------------------------------------ NER models
def test_crf_features():
    assert word_shape("E330") == "Xdd" and word_shape("Sugar") == "Xxx" and word_shape("INS") == "XX"
    feats = sentence_features(["Sugar", ",", "INS", "330"])
    assert feats[3]["is_digit"] and feats[3]["-1:lower"] == "ins" and "1:edge" in feats[3]


def test_noisy_copy_keeps_alignment():
    record = {"text": "Sugar, Salt, INS 330", "tokens": ["Sugar", ",", "Salt", ",", "INS", "330"],
              "token_offsets": [[0, 5], [5, 6], [7, 11], [11, 12], [13, 16], [17, 20]],
              "entities": [{"start": 0, "end": 5, "text": "Sugar", "label": "SUGAR"}]}
    noisy = noisy_copy([record], 1.0)[0]
    assert len(noisy["text"]) == len(record["text"]) and noisy["text"] != record["text"]
    assert noisy["tokens"] == [noisy["text"][s:e] for s, e in record["token_offsets"]]


def test_subword_label_alignment():
    transformers = pytest.importorskip("transformers")
    vocab = project_path("models/pretrained/distilbert-base-uncased")
    if not (vocab / "vocab.txt").exists():
        pytest.skip("pretrained tokenizer not downloaded")
    from src.ner.transformer_ner import IGNORE, encode, fix_bio
    from src.labeling.bio import LABEL2ID
    tokenizer = transformers.AutoTokenizer.from_pretrained(vocab)
    item = encode(tokenizer, ["Maltodextrin", "INS", "330"], ["B-INGREDIENT", "B-INS_CODE", "I-INS_CODE"])
    labelled = [l for l in item["labels"] if l != IGNORE]
    assert labelled == [LABEL2ID["B-INGREDIENT"], LABEL2ID["B-INS_CODE"], LABEL2ID["I-INS_CODE"]]
    assert len(item["first_subword"]) == 3
    assert fix_bio(["O", "I-SUGAR", "I-SUGAR", "I-FAT"]) == ["O", "B-SUGAR", "I-SUGAR", "B-FAT"]


def test_exported_model_predicts():
    model_dir = project_path("models/ingredient-ner-distilbert")
    if not (model_dir / "config.json").exists():
        pytest.skip("fine-tuned model not exported yet")
    from src.ner.transformer_ner import TransformerTagger
    out = TransformerTagger(model_dir).predict("Sugar, glucose syrup, Acidity regulator (INS 330), palm oil")
    labels = [e["label"] for e in out["entities"]]
    assert "SUGAR" in labels and "INS_CODE" in labels and "FAT" in labels
