"""Error analysis: compare gold entities with predicted entities and explain every difference.

Usage:  python -m src.evaluation.error_analysis      (needs data/splits/gold/test.jsonl)

Error types (one row per error in reports/error_analysis/baseline_errors.csv):
  false_negative      a gold entity the system did not find at all (no overlapping prediction)
  false_positive      a predicted entity that overlaps no gold entity
  boundary_error      same label, spans overlap but are not identical ("cane sugar" vs "organic cane sugar")
  type_error          same span, different label ("citric acid": ADDITIVE vs INGREDIENT)
  boundary_and_type   overlapping span AND different label

Extra tags explain WHY (several can apply):
  ins_code_variant    an INS/E code is involved (notation such as "INS-330", "E 150 d", "(330)")
  ocr_noise           a word in the entity is not in the corpus vocabulary (likely OCR/spelling error)
  compound            the entity is inside brackets (a sub-ingredient of a compound ingredient)
  unknown_term        a false negative whose text is in none of the dictionaries
  overlapping_category the two labels are both additive-type labels (e.g. PRESERVATIVE vs ADDITIVE)
  ambiguous_term      the text is listed as ambiguous in the annotation guideline
"""
import re

import pandas as pd

from src.labeling.lexicon import load_lexicon, lookup_key
from src.labeling.weak_labeler import corpus_vocabulary
from src.utils.config import project_path

ADDITIVE_TYPES = {"ADDITIVE", "PRESERVATIVE", "COLOUR", "SWEETENER", "INS_CODE"}
AMBIGUOUS_TERMS = {"maltodextrin", "citric acid", "ascorbic acid", "lactose", "caramel", "colour", "color",
                   "natural colour", "cocoa butter", "milk fat", "cream", "vanilla", "vanilla extract",
                   "fruit juice concentrate", "calcium carbonate", "lecithin", "glycerol", "sorbitol",
                   "modified starch", "paprika extract", "turmeric"}
CONTEXT = 40


def _depth(text: str, pos: int) -> int:
    before = text[:pos]
    return sum(before.count(c) for c in "([{") - sum(before.count(c) for c in ")]}")


def _tags(text, gold, pred, error_type) -> list:
    tags = []
    ent = gold or pred
    labels = {e["label"] for e in (gold, pred) if e}
    if "INS_CODE" in labels or re.search(r"\b(?:E|INS)\s?-?\s?\d{3}|\(\d{3}", ent["text"], re.IGNORECASE):
        tags.append("ins_code_variant")
    vocabulary = corpus_vocabulary()
    if vocabulary and any(w.lower() not in vocabulary for w in re.findall(r"[^\W\d_]{3,}", ent["text"])):
        tags.append("ocr_noise")
    if _depth(text, ent["start"]) > 0:
        tags.append("compound")
    if error_type == "false_negative" and lookup_key(gold["text"]) not in load_lexicon():
        tags.append("unknown_term")
    if len(labels) == 2 and labels <= ADDITIVE_TYPES:
        tags.append("overlapping_category")
    if ent["text"].lower() in AMBIGUOUS_TERMS:
        tags.append("ambiguous_term")
    return tags


def compare_entities(text: str, gold: list, pred: list) -> list:
    """Return a list of error dicts for one document."""
    errors, matched_pred = [], set()
    pred_keys = {(p["start"], p["end"], p["label"]) for p in pred}
    for g in gold:
        if (g["start"], g["end"], g["label"]) in pred_keys:
            matched_pred.add((g["start"], g["end"], g["label"]))
            continue
        overlapping = [p for p in pred if p["start"] < g["end"] and g["start"] < p["end"]]
        if not overlapping:
            errors.append(("false_negative", g, None))
            continue
        p = max(overlapping, key=lambda p: min(p["end"], g["end"]) - max(p["start"], g["start"]))
        matched_pred.add((p["start"], p["end"], p["label"]))
        same_span = (p["start"], p["end"]) == (g["start"], g["end"])
        if p["label"] == g["label"]:
            errors.append(("boundary_error", g, p))
        elif same_span:
            errors.append(("type_error", g, p))
        else:
            errors.append(("boundary_and_type", g, p))
    for p in pred:
        key = (p["start"], p["end"], p["label"])
        if key not in matched_pred and not any(p["start"] < g["end"] and g["start"] < p["end"] for g in gold):
            errors.append(("false_positive", None, p))

    rows = []
    for error_type, g, p in errors:
        ent = g or p
        rows.append({
            "text": text[max(0, ent["start"] - CONTEXT):ent["end"] + CONTEXT],
            "expected_entity": g["text"] if g else "",
            "expected_label": g["label"] if g else "",
            "predicted_entity": p["text"] if p else "",
            "predicted_label": p["label"] if p else "",
            "error_type": error_type,
            "entity_category": g["label"] if g else p["label"],
            "tags": "|".join(_tags(text, g, p, error_type)),
        })
    return rows


def error_table(gold_records: list, predictions: list) -> pd.DataFrame:
    rows = []
    for record, pred in zip(gold_records, predictions):
        for row in compare_entities(record["text"], record["entities"], pred):
            rows.append({"product_id": record["product_id"], **row})
    return pd.DataFrame(rows, columns=["product_id", "text", "expected_entity", "expected_label", "predicted_entity",
                                       "predicted_label", "error_type", "entity_category", "tags"])


def summarise(errors: pd.DataFrame) -> dict:
    tags = errors["tags"].str.split("|").explode()
    return {
        "errors": len(errors),
        "by_error_type": errors["error_type"].value_counts().to_dict(),
        "by_category": errors["entity_category"].value_counts().to_dict(),
        "by_tag": tags[tags != ""].value_counts().to_dict(),
        "type_confusions": errors[errors["error_type"].isin(["type_error", "boundary_and_type"])]
        .groupby(["expected_label", "predicted_label"]).size().sort_values(ascending=False).head(15)
        .rename(lambda x: f"{x[0]} -> {x[1]}").to_dict(),
    }


def main():
    from src.evaluation.evaluate_baseline import load_gold_or_exit, predict_all
    import json
    gold = load_gold_or_exit("test")
    errors = error_table(gold, predict_all(gold))
    out = project_path("reports/error_analysis/baseline_errors.csv")
    errors.to_csv(out, index=False)
    summary = summarise(errors)
    project_path("reports/error_analysis/baseline_error_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{len(errors)} errors -> {out.relative_to(project_path(''))}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
