"""Tests for the weak labeller, lexicon, BIO conversion, interpretation and validation."""
import pytest

from src.labeling.bio import BIO_TAGS, ID2LABEL, LABEL2ID, LABELS, bio_to_entities, entities_to_bio
from src.labeling.interpret import interpret_entities
from src.labeling.lexicon import label_for_number, load_lexicon, lookup_key
from src.labeling.weak_labeler import label_text
from src.utils.validation import DataValidationError, check_bio_sequence, validate_records


def labelled(text):
    return [(e["text"], e["label"]) for e in label_text(text)["entities"]]


def test_schema_has_ten_labels_and_consistent_ids():
    assert len(LABELS) == 10 and {"PRESERVATIVE", "COLOUR"} <= set(LABELS)
    assert len(BIO_TAGS) == 21 and LABEL2ID["O"] == 0
    assert all(ID2LABEL[i] == tag for tag, i in LABEL2ID.items())


def test_ins_number_rules():
    assert label_for_number("102") == "COLOUR"
    assert label_for_number("211") == "PRESERVATIVE"
    assert label_for_number("270") == "ADDITIVE"        # lactic acid: acid, excluded from preservatives
    assert label_for_number("170") == "ADDITIVE"        # calcium carbonate: excluded from colours
    assert label_for_number("955") == "SWEETENER"
    assert label_for_number("330") == "ADDITIVE"


def test_lexicon_policy_decisions():
    lex = load_lexicon()
    assert "riboflavin" not in lex and "niacin" not in lex         # vitamins are INGREDIENT, not additives
    assert lex["sucralose"].number == "955"
    assert lookup_key("Mono - and Diglycerides") == lookup_key("mono- and diglycerides")
    assert lookup_key("Yellow #6") == "yellow 6"


def test_example_from_the_brief():
    assert labelled("sugar, glucose syrup, INS 330, palm oil") == [
        ("sugar", "SUGAR"), ("glucose syrup", "SUGAR"), ("INS 330", "INS_CODE"), ("palm oil", "FAT")]


def test_function_class_plus_code():
    assert labelled("Acidity regulator (INS 330), Colour (150d)") == [
        ("Acidity regulator", "FUNCTION_CLASS"), ("INS 330", "INS_CODE"), ("Colour", "FUNCTION_CLASS"),
        ("150d", "INS_CODE")]


def test_preservative_and_colour_labels():
    assert labelled("potassium benzoate and potassium sorbate (preservatives), tartrazine, red 40") == [
        ("potassium benzoate", "PRESERVATIVE"), ("potassium sorbate", "PRESERVATIVE"),
        ("preservatives", "FUNCTION_CLASS"), ("tartrazine", "COLOUR"), ("red 40", "COLOUR")]


def test_head_word_rules_and_exceptions():
    assert labelled("refined palm oil, organic cane sugar, peanut butter, spearmint oil, maltitol syrup") == [
        ("refined palm oil", "FAT"), ("organic cane sugar", "SUGAR"), ("peanut butter", "INGREDIENT"),
        ("spearmint oil", "FLAVOURING"), ("maltitol syrup", "SWEETENER")]


def test_framing_and_statements_are_not_entities():
    assert labelled("Sugar; contains less than 2% of: salt. Contains: milk, soy. May contain nuts.") == [
        ("Sugar", "SUGAR"), ("salt", "INGREDIENT")]
    assert labelled("Sugar (20.5%), natural flavour (to protect flavour)") == [
        ("Sugar", "SUGAR"), ("natural flavour", "FLAVOURING")]


def test_class_name_followed_by_substance():
    assert labelled("Emulsifier soy lecithin") == [("Emulsifier", "FUNCTION_CLASS"), ("soy lecithin", "ADDITIVE")]


def test_bio_round_trip_and_alignment_errors():
    result = label_text("Acidity regulator (INS 330), salt")
    tags = entities_to_bio(result["token_offsets"], result["entities"])
    assert tags == ["B-FUNCTION_CLASS", "I-FUNCTION_CLASS", "O", "B-INS_CODE", "I-INS_CODE", "O", "O", "B-INGREDIENT"]
    back = bio_to_entities(result["token_offsets"], tags, result["text"])
    assert [(e["text"], e["label"]) for e in back] == [(e["text"], e["label"]) for e in result["entities"]]
    with pytest.raises(ValueError):                     # an entity boundary inside a token must fail loudly
        entities_to_bio([(0, 5)], [{"start": 0, "end": 3, "label": "SUGAR", "text": "Sug"}])


def test_interpretation_without_health_claims():
    result = label_text("Acidity regulator (INS 330), Emulsifiers (471, soy lecithin), sodium benzoate (preservative)")
    ents = {e["text"]: e for e in interpret_entities(result["text"], result["entities"])}
    assert ents["INS 330"]["linked_name"] == "Citric acid"
    assert ents["INS 330"]["declared_class"] == "acidity-regulator"
    assert ents["soy lecithin"]["declared_class"] == "emulsifier"
    assert ents["sodium benzoate"]["declared_class"] == "preservative"


def test_validation_detects_malformed_records():
    good = label_text("Sugar, salt")
    record = {"id": "1", "text": good["text"], "tokens": good["tokens"], "token_offsets": good["token_offsets"],
              "ner_tags": entities_to_bio(good["token_offsets"], good["entities"]), "entities": good["entities"]}
    assert validate_records([record])["problems"] == []
    assert check_bio_sequence(["O", "I-SUGAR"]) == ["I-SUGAR at position 1 follows O"]
    assert check_bio_sequence(["B-SUGAR", "I-FAT"]) == ["I-FAT at position 1 follows B-SUGAR"]
    bad = {**record, "ner_tags": record["ner_tags"][:-1]}
    with pytest.raises(DataValidationError):
        validate_records([bad])
    with pytest.raises(DataValidationError):
        validate_records([record, record])            # duplicate ids
    with pytest.raises(DataValidationError):
        validate_records([{**record, "ner_tags": ["B-NOT_A_LABEL"] + record["ner_tags"][1:]}])
    with pytest.raises(DataValidationError):
        validate_records([{**record, "text": " ", "tokens": []}])


def test_phrase_rules_found_in_real_data():
    assert labelled("mixed tocopherols added to preserve freshness, BHT for freshness, sugar for dusting") == [
        ("mixed tocopherols", "ADDITIVE"), ("BHT", "ADDITIVE"), ("sugar", "SUGAR")]
    assert labelled("Emulsifier of vegetable origin, rapeseed oil in varying proportions") == [
        ("Emulsifier", "FUNCTION_CLASS"), ("rapeseed oil", "FAT")]
    assert labelled("I want to know more. Packaged in a protective atmosphere. Salt") == [("Salt", "INGREDIENT")]
    assert labelled("Bhendi") == [("Bhendi", "INGREDIENT")]           # single-ingredient list, rare word
    assert labelled("Sugar, acd anin b-12") == [("Sugar", "SUGAR")]    # OCR garbage is not labelled
