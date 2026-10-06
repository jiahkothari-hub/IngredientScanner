"""Export the final processed dataset as CSV (one row per product, readable in Excel / pandas).

Usage:  python -m src.data.export_processed       (after build_silver and split_dataset)

Outputs:
  data/processed/ingredients_processed.csv   one row per product: metadata, split, raw + normalised
                                             text, and the detected entities grouped by label
  data/processed/silver_entities.csv.gz      one row per entity, with interpretation (linked name,
                                             declared and reference functional classes)

All entity columns come from the SILVER (automatic) labels - see the `label_source` column.
List columns use " | " as separator.
"""
import pandas as pd

from src.labeling.bio import LABELS
from src.labeling.interpret import describe, interpret_entities
from src.utils.config import load_config, project_path
from src.utils.io import read_jsonl

COLUMN_FOR_LABEL = {
    "SUGAR": "sugars", "SWEETENER": "sweeteners", "FAT": "fats", "PRESERVATIVE": "preservatives",
    "COLOUR": "colours", "ADDITIVE": "additives", "INS_CODE": "ins_codes", "FUNCTION_CLASS": "function_classes",
    "FLAVOURING": "flavourings", "INGREDIENT": "other_ingredients",
}


def unique_join(values) -> str:
    return " | ".join(dict.fromkeys(v for v in values if v))


def main():
    config = load_config()
    products = pd.read_csv(project_path(config["outputs"]["products"]), dtype={"product_id": str})
    split = pd.read_csv(project_path("data/processed/split_assignment.csv"), dtype={"product_id": str})
    meta = products.set_index("product_id")
    split_of = dict(zip(split["product_id"], split["split"]))

    product_rows, entity_rows = [], []
    for r in read_jsonl(project_path("data/processed/silver.jsonl.gz")):
        entities = interpret_entities(r["text"], r["entities"])
        m = meta.loc[r["product_id"]]
        row = {
            "product_id": r["product_id"],
            "product_name": m["product_name"],
            "brands": m["brands"],
            "country_group": r["country_group"],
            "main_category": m["main_category"],
            "split": split_of.get(r["product_id"], ""),
            "ingredients_text_raw": r["text_raw"],
            "ingredients_text_normalized": r["text"],
            "n_tokens": len(r["tokens"]),
            "n_entities": len(entities),
        }
        for label in LABELS:
            row[COLUMN_FOR_LABEL[label]] = unique_join(e["text"] for e in entities if e["label"] == label)
        row["ins_numbers"] = unique_join(e["number"] for e in entities if e["label"] == "INS_CODE")
        row["additive_interpretations"] = unique_join(
            describe(e) for e in entities if e.get("linked_name") or e.get("declared_class"))
        for label in LABELS:
            row[f"n_{label}"] = sum(e["label"] == label for e in entities)
        row["label_source"] = r["label_source"]
        product_rows.append(row)

        for i, e in enumerate(entities):
            entity_rows.append({
                "product_id": r["product_id"], "split": row["split"], "entity_index": i,
                "start": e["start"], "end": e["end"], "text": e["text"], "label": e["label"], "rule": e["rule"],
                "ins_number": e.get("number", ""), "linked_name": e.get("linked_name", ""),
                "declared_class": e.get("declared_class", ""),
                "reference_classes": " | ".join(e.get("reference_classes", [])),
                "interpretation": describe(e),
            })

    product_df = pd.DataFrame(product_rows)
    product_df.to_csv(project_path("data/processed/ingredients_processed.csv"), index=False)
    pd.DataFrame(entity_rows).to_csv(project_path("data/processed/silver_entities.csv.gz"), index=False,
                                     compression={"method": "gzip", "mtime": 0})  # mtime=0: identical bytes every run
    print(f"Saved {len(product_df):,} products -> data/processed/ingredients_processed.csv")
    print(f"Saved {len(entity_rows):,} entities -> data/processed/silver_entities.csv.gz")


if __name__ == "__main__":
    main()
