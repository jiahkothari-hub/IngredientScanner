"""Agreement between our additive detection and Open Food Facts' own additive parser.

Usage:  python -m src.evaluation.off_agreement

This is a SANITY CHECK that needs no gold data. For every product in the silver TEST split we
compare two sets of additive numbers:
  ours = numbers of INS_CODE entities + numbers linked to named additives (e.g. "citric acid" -> 330)
  OFF  = numbers in the product's `additives_tags` column (e.g. "en:e330")
Numbers are compared at base level (322i -> 322) because the two systems resolve sub-types differently.

  agreement precision = |ours & OFF| / |ours|     recall = |ours & OFF| / |OFF|

Caveat for the report: OFF's parser uses the same taxonomy names, so the two systems are not fully
independent - this is NOT an accuracy measurement. It shows where the systems disagree, which is
useful for error analysis (e.g. bare numbers "(471)" that OFF misses, or names we do not cover).
"""
import json
import re
from collections import Counter

import pandas as pd

from src.utils.config import load_config, project_path
from src.utils.io import read_jsonl


def base(number: str) -> str:
    """'322i' -> '322'. Modified starches (E1400-E1452) are compared as one family '14xx', because
    OFF often tags them generically as 'e14xx' while labels name a specific one."""
    if number.startswith("14") and (len(number) >= 4 or number.lower() == "14xx"):
        return "14xx"
    return re.match(r"\d+", number).group(0)


def our_numbers(record: dict) -> set:
    return {base(e["number"]) for e in record["entities"] if e.get("number")}


def off_numbers(tags) -> set:
    if not isinstance(tags, str):
        return set()
    return {base(t[4:]) for t in tags.split(",") if re.match(r"en:e\d", t)}  # "en:e330" -> "330"


def main():
    config = load_config()
    products = pd.read_csv(project_path(config["outputs"]["products"]), dtype={"product_id": str})
    off = dict(zip(products["product_id"], products["off_additives_tags"]))
    records = read_jsonl(project_path("data/splits/silver/test.jsonl.gz"))

    tp = n_ours = n_off = 0
    only_ours, only_off = Counter(), Counter()
    for r in records:
        ours, theirs = our_numbers(r), off_numbers(off.get(r["product_id"]))
        tp += len(ours & theirs)
        n_ours += len(ours)
        n_off += len(theirs)
        only_ours.update(ours - theirs)
        only_off.update(theirs - ours)
    precision, recall = tp / n_ours, tp / n_off
    result = {
        "products": len(records),
        "additive_numbers_ours": n_ours, "additive_numbers_off": n_off, "shared": tp,
        "agreement_precision": round(precision, 4),
        "agreement_recall": round(recall, 4),
        "agreement_f1": round(2 * precision * recall / (precision + recall), 4),
        "most_common_only_ours": dict(only_ours.most_common(15)),
        "most_common_only_off": dict(only_off.most_common(15)),
    }
    project_path("reports/off_additive_agreement.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
