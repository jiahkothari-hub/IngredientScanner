"""Select products for human GOLD annotation and export them for Doccano.

Usage:  python -m src.annotation.sample_gold        (after src.data.split_dataset)

Selection (per split, so the gold TEST set comes only from test groups and stays held out):
  1. Enrichment: a fixed share of hard / rare cases (with INS codes, preservatives, colours,
     sweeteners, several sugars, several fats, noisy OCR-like text) - configs/data_config.yaml.
  2. The rest: stratified random sample by market group x length bucket (short / medium / long).
  3. 40 extra train products are given to ALL annotators to measure inter-annotator agreement.
Every product records WHY it was selected (selection_reason).

Outputs (data/annotations/):
  batches/<annotator>.jsonl     Doccano import files, PRE-ANNOTATED with silver labels
  doccano_label_config.json     the 10 entity types with colours and shortcut keys for Doccano
  annotation_tracking.csv       who annotates what, status, notes
IMPORTANT: these files are NOT gold. They become gold only after a person has checked and corrected
every entity and the result is imported with src/annotation/import_annotations.py.
"""
import json
import random
import re

import pandas as pd

from src.labeling.bio import LABELS
from src.labeling.weak_labeler import corpus_vocabulary
from src.utils.config import load_config, project_path
from src.utils.io import read_jsonl, write_jsonl

ANNOTATION_DIR = project_path("data/annotations")
COLOURS = {"SUGAR": "#eb6834", "SWEETENER": "#e87ba4", "FAT": "#eda100", "PRESERVATIVE": "#e34948",
           "COLOUR": "#4a3aa7", "ADDITIVE": "#2a78d6", "INS_CODE": "#1baf7a", "FUNCTION_CLASS": "#52514e",
           "FLAVOURING": "#008300", "INGREDIENT": "#898781"}
SHORTCUTS = {"SUGAR": "s", "SWEETENER": "w", "FAT": "f", "PRESERVATIVE": "p", "COLOUR": "c",
             "ADDITIVE": "a", "INS_CODE": "n", "FUNCTION_CLASS": "k", "FLAVOURING": "v", "INGREDIENT": "i"}


def record_features(record: dict) -> dict:
    labels = [e["label"] for e in record["entities"]]
    words = re.findall(r"[^\W\d_]{3,}", record["text"].lower())
    vocabulary = corpus_vocabulary()
    unknown = sum(w not in vocabulary for w in words) / len(words) if words and vocabulary else 0.0
    n = len(record["tokens"])
    return {
        "has_ins_code": "INS_CODE" in labels,
        "has_preservative": "PRESERVATIVE" in labels,
        "has_colour": "COLOUR" in labels,
        "has_sweetener": "SWEETENER" in labels,
        "multiple_sugars": labels.count("SUGAR") >= 2,
        "multiple_fats": labels.count("FAT") >= 2,
        "noisy_text": unknown >= 0.15,                       # many words never seen in other products
        "length_bucket": "short" if n <= 25 else "medium" if n <= 120 else "long",
    }


def stratified_sample(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Proportional allocation over market x length strata (largest-remainder rounding)."""
    if n <= 0 or pool.empty:
        return pool.iloc[0:0]
    strata = pool.groupby(["country_group", "length_bucket"])
    sizes = strata.size()
    exact = sizes / sizes.sum() * n
    alloc = exact.astype(int)
    for key in (exact - alloc).sort_values(ascending=False).index[: n - alloc.sum()]:
        alloc[key] += 1
    parts = [group.sample(n=min(alloc[key], len(group)), random_state=seed) for key, group in strata if alloc[key]]
    return pd.concat(parts)


def select_for_split(pool: pd.DataFrame, n: int, enrich: dict, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    chosen, reasons = [], {}
    for feature, share in enrich.items():
        candidates = sorted(set(pool.index[pool[feature]]) - set(chosen))
        picks = rng.sample(candidates, min(len(candidates), round(share * n)))
        chosen += picks
        reasons.update({i: feature for i in picks})
    rest = stratified_sample(pool.drop(index=chosen), n - len(chosen), seed)
    reasons.update({i: "stratified_random" for i in rest.index})
    out = pool.loc[chosen + list(rest.index)].copy()
    out["selection_reason"] = out.index.map(reasons)
    return out


def main():
    config = load_config()
    gcfg, seed = config["gold"], config["random_seed"]
    records = {r["id"]: r for r in read_jsonl(project_path("data/processed/silver.jsonl.gz"))}
    assignment = pd.read_csv(project_path("data/processed/split_assignment.csv"), dtype={"product_id": str})

    features = pd.DataFrame([{"product_id": pid, **record_features(r)} for pid, r in records.items()])
    pool = assignment.merge(features, on="product_id").set_index("product_id", drop=False)

    selected = []
    for split, n in gcfg["sizes"].items():
        part = select_for_split(pool[pool["split"] == split], n, gcfg["enrich"], seed)
        selected.append(part)
    selected = pd.concat(selected)

    # inter-annotator agreement items: random train items, annotated by everybody
    train_ids = sorted(selected.index[selected["split"] == "train"])
    iaa = set(random.Random(seed).sample(train_ids, gcfg["agreement_items"]))
    selected["agreement_item"] = selected.index.isin(iaa)

    # assign the remaining items round-robin (in a shuffled but reproducible order)
    annotators = gcfg["annotators"]
    others = sorted(set(selected.index) - iaa)
    random.Random(seed).shuffle(others)
    assigned = {pid: annotators[i % len(annotators)] for i, pid in enumerate(others)}
    selected["assigned_to"] = [("ALL" if pid in iaa else assigned[pid]) for pid in selected.index]

    batch_dir = ANNOTATION_DIR / "batches"
    for annotator in annotators:
        mine = selected[(selected["assigned_to"] == annotator) | selected["agreement_item"]]
        lines = []
        for pid, row in mine.sort_values(["agreement_item", "split"], ascending=[False, True]).iterrows():
            r = records[pid]
            lines.append({
                "text": r["text"],
                "label": [[e["start"], e["end"], e["label"]] for e in r["entities"]],
                "product_id": pid, "split": row["split"], "country_group": row["country_group"],
                "selection_reason": row["selection_reason"], "agreement_item": bool(row["agreement_item"]),
                "pre_annotation": "silver (unverified) - check every entity",
            })
        write_jsonl(lines, batch_dir / f"{annotator}.jsonl")

    label_config = [{"text": label, "suffix_key": SHORTCUTS[label], "background_color": COLOURS[label],
                     "text_color": "#ffffff"} for label in LABELS]
    (ANNOTATION_DIR / "doccano_label_config.json").write_text(json.dumps(label_config, indent=2), encoding="utf-8")

    tracking = selected[["product_id", "split", "country_group", "selection_reason", "agreement_item",
                         "assigned_to"]].reset_index(drop=True)
    tracking["status"] = "to_do"
    tracking["verified_by"] = ""
    tracking["notes"] = ""
    tracking.sort_values(["assigned_to", "split", "product_id"]).to_csv(
        ANNOTATION_DIR / "annotation_tracking.csv", index=False)

    print(f"Selected {len(selected)} products: {selected['split'].value_counts().to_dict()}")
    print("Selection reasons:", selected["selection_reason"].value_counts().to_dict())
    print("Per annotator (incl. agreement items):",
          {a: int(((selected['assigned_to'] == a) | selected['agreement_item']).sum()) for a in annotators})


if __name__ == "__main__":
    main()
