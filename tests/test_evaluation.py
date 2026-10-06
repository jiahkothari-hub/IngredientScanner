"""Tests for metrics, error analysis, baseline, gold import and the split logic.
All gold-like data here are small hand-written TEST FIXTURES, not project annotations."""
import json

from src.annotation.agreement import agreement
from src.annotation.import_annotations import build_gold, load_exports, parse_doccano_line
from src.baseline.dictionary_ner import DictionaryNER
from src.data.split_dataset import UnionFind, assign_splits
from src.evaluation.error_analysis import compare_entities
from src.evaluation.metrics import evaluate_entities, evaluate_tag_sequences, token_cohen_kappa
from src.evaluation.noise_robustness import add_ocr_noise


def ent(start, end, label, text=""):
    return {"start": start, "end": end, "label": label, "text": text}


def test_entity_level_metrics():
    gold = [[ent(0, 5, "SUGAR"), ent(7, 15, "FAT"), ent(17, 24, "INS_CODE")]]
    pred = [[ent(0, 5, "SUGAR"), ent(7, 15, "INGREDIENT"), ent(17, 22, "INS_CODE"), ent(30, 35, "SUGAR")]]
    r = evaluate_entities(gold, pred)
    assert r["micro"]["correct"] == 1 and r["micro"]["precision"] == 0.25 and r["micro"]["recall"] == round(1 / 3, 4)
    assert r["per_class"]["SUGAR"]["precision"] == 0.5 and r["per_class"]["SUGAR"]["recall"] == 1.0
    assert r["partial_micro"]["correct"] == 2            # the shortened INS_CODE counts as partial


def test_tag_sequence_metrics_and_kappa():
    gold = [["B-SUGAR", "O", "B-FAT", "I-FAT"]]
    assert evaluate_tag_sequences(gold, gold)["micro"]["f1"] == 1.0
    assert evaluate_tag_sequences(gold, [["B-SUGAR", "O", "B-FAT", "O"]])["micro"]["f1"] == 0.5
    assert token_cohen_kappa(gold, gold) == 1.0


def test_error_types():
    text = "organic cane sugar, citric acid, salt, xyz"
    gold = [ent(0, 18, "SUGAR", "organic cane sugar"), ent(20, 31, "ADDITIVE", "citric acid"),
            ent(33, 37, "INGREDIENT", "salt")]
    pred = [ent(8, 18, "SUGAR", "cane sugar"), ent(20, 31, "INGREDIENT", "citric acid"),
            ent(39, 42, "INGREDIENT", "xyz")]
    types = sorted(row["error_type"] for row in compare_entities(text, gold, pred))
    assert types == ["boundary_error", "false_negative", "false_positive", "type_error"]


def test_baseline_predicts_on_normalised_records():
    ner = DictionaryNER()
    out = ner.predict("Sugar, INS 330")
    record = {"id": "x", "text": out["text"], "token_offsets": out["token_offsets"]}
    assert [e["label"] for e in ner.predict_record(record)] == ["SUGAR", "INS_CODE"]
    assert ner.predict_tags(record) == ["B-SUGAR", "O", "B-INS_CODE", "I-INS_CODE"]


def test_ocr_noise_keeps_length():
    import random
    text = "Sugar, Salt, INS 330, palm oil"
    noisy = add_ocr_noise(text, 1.0, random.Random(0))
    assert len(noisy) == len(text) and noisy != text


def test_union_find_and_group_split():
    uf = UnionFind(["a", "b", "c"])
    uf.union("a", "b")
    assert uf.find("a") == uf.find("b") != uf.find("c")
    records = [{"id": str(i), "country_group": "india"} for i in range(100)]
    group_of = {str(i): str(i // 2) for i in range(100)}          # 50 groups of 2 products
    split_of = assign_splits(records, group_of, seed=1)
    for i in range(0, 100, 2):
        assert split_of[str(i)] == split_of[str(i + 1)]           # a group never crosses splits
    counts = {s: list(split_of.values()).count(s) for s in ("train", "validation", "test")}
    assert counts == {"train": 70, "validation": 14, "test": 16} or abs(counts["train"] - 70) <= 2


def test_doccano_import_and_agreement(tmp_path):
    """Two fake annotators (fixture), both Doccano export formats, one disagreement."""
    text = "Sugar, Acidity regulator (INS 330)"
    a = {"text": text, "product_id": "p1", "label": [[0, 5, "SUGAR"], [7, 24, "FUNCTION_CLASS"], [26, 33, "INS_CODE"]]}
    b = {"text": text, "product_id": "p1", "entities": [{"start_offset": 0, "end_offset": 6, "label": "SUGAR"},
                                                        {"start_offset": 26, "end_offset": 33, "label": "INS_CODE"}]}
    c = {"text": "Salt", "product_id": "p2", "label": [[0, 4, "INGREDIENT"]]}
    (tmp_path / "ann_a.jsonl").write_text(json.dumps(a) + "\n" + json.dumps(c) + "\n")
    (tmp_path / "ann_b.jsonl").write_text(json.dumps(b) + "\n")
    assert parse_doccano_line(b)[0] == (0, 6, "SUGAR")

    silver = tmp_path / "silver.jsonl"
    silver.write_text("")
    records, warnings, unresolved = build_gold(export_dir=tmp_path, silver_path=silver)
    assert unresolved == ["p1"]                          # disagreement -> not gold until adjudicated
    assert [r["id"] for r in records] == ["p2"] and records[0]["status"] == "verified"

    (tmp_path / "adjudicated.jsonl").write_text(json.dumps(a) + "\n")
    records, warnings, unresolved = build_gold(export_dir=tmp_path, silver_path=silver)
    assert unresolved == [] and records[0]["annotators"][-1] == "adjudicated"
    assert records[0]["entities"][0]["text"] == "Sugar"  # span "Sugar," trimmed to the token

    pair = agreement(load_exports(tmp_path))["ann_a vs ann_b"]
    assert pair["documents"] == 1 and pair["entity_f1"] == round(2 * 0.5 * (1 / 3) / (0.5 + 1 / 3), 4)
