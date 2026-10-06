"""Create the SILVER dataset: every product labelled automatically by the weak labeller.

Usage:  python -m src.labeling.build_silver      (after build_vocabulary)

SILVER = automatically generated. It is a large training resource, NOT ground truth.

Each output line (data/processed/silver.jsonl.gz) looks like:
{
  "id": "8901058000290", "product_id": "8901058000290", "country_group": "india",
  "text_raw": "...original text...", "text": "...normalised text...",
  "tokens": ["Sugar", ",", ...], "token_offsets": [[0, 5], ...], "ner_tags": ["B-SUGAR", "O", ...],
  "entities": [{"start": 0, "end": 5, "text": "Sugar", "label": "SUGAR", "rule": "dict:sugars",
                "confidence": null}, ...],
  "label_source": "silver_weak_supervision", "labeler_version": "1.0"
}
`rule` says which dictionary/regex produced the entity. `confidence` stays null until it is
measured: src/evaluation/rule_precision.py computes each rule's precision on human-verified gold
data and writes it here. We never invent confidence values.
"""
import json
from collections import Counter

import pandas as pd

from src.labeling.bio import entities_to_bio
from src.labeling.weak_labeler import label_text
from src.utils.config import load_config, project_path
from src.utils.io import write_jsonl
from src.utils.validation import MAX_TOKENS, validate_records

LABELER_VERSION = "1.0"
MAX_NON_LATIN_RATIO = 0.2   # bilingual labels with a large Arabic/other-script part are skipped
SILVER_PATH = project_path("data/processed/silver.jsonl.gz")
LOG_PATH = project_path("reports/silver_log.json")


def make_record(product: dict) -> dict:
    """Weak-label one product row; returns a silver record (or raises ValueError if unusable)."""
    result = label_text(product["ingredients_text_raw"])
    if not result["tokens"]:
        raise ValueError("empty after normalisation")
    if result["non_latin_ratio"] > MAX_NON_LATIN_RATIO:
        raise ValueError("mostly non-Latin script")
    if len(result["tokens"]) > MAX_TOKENS:
        raise ValueError("too many tokens")
    entities = [{**ent, "confidence": None} for ent in result["entities"]]
    return {
        "id": product["product_id"],
        "product_id": product["product_id"],
        "country_group": product["country_group"],
        "text_raw": result["raw_text"],
        "text": result["text"],
        "tokens": result["tokens"],
        "token_offsets": result["token_offsets"],
        "ner_tags": entities_to_bio(result["token_offsets"], entities),
        "entities": entities,
        "additive_codes": [c["number"] for c in result["additive_codes"]],
        "label_source": "silver_weak_supervision",
        "labeler_version": LABELER_VERSION,
    }


def main():
    config = load_config()
    products = pd.read_csv(project_path(config["outputs"]["products"]), dtype={"product_id": str})
    records, skipped = [], Counter()
    for product in products.to_dict("records"):
        try:
            records.append(make_record(product))
        except ValueError as exc:
            skipped[str(exc)] += 1

    report = validate_records(records, name="silver")          # raises if anything is malformed
    write_jsonl(records, SILVER_PATH)

    label_counts = Counter(e["label"] for r in records for e in r["entities"])
    rule_counts = Counter(e["rule"] for r in records for e in r["entities"])
    log = {
        "products_in": len(products),
        "records_out": len(records),
        "skipped": dict(skipped),
        "entities": sum(label_counts.values()),
        "entities_per_label": dict(label_counts.most_common()),
        "entities_per_rule": dict(rule_counts.most_common()),
        "tokens": sum(len(r["tokens"]) for r in records),
        "tokens_tagged_O_percent": round(100 * sum(t == "O" for r in records for t in r["ner_tags"])
                                         / sum(len(r["tokens"]) for r in records), 1),
        "inconsistent_surface_forms": report["inconsistent_surface_forms"],
    }
    LOG_PATH.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"Silver: {len(records):,} records, {log['entities']:,} entities, skipped {dict(skipped)}")
    print(f"Validation passed. Saved {SILVER_PATH.relative_to(project_path(''))}")


if __name__ == "__main__":
    main()
