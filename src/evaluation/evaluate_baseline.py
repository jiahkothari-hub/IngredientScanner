"""Evaluate the dictionary/rule baseline on the human-verified GOLD test set.

Usage:  python -m src.evaluation.evaluate_baseline [--split test|validation]

Outputs:
  reports/baseline_metrics.json / .md     overall + per-class Precision, Recall, F1
  reports/figures/09_baseline_per_class_f1.png
  reports/figures/10_baseline_label_confusion.png   which gold labels become which predicted labels

Without gold annotations this script stops with a clear message: the baseline is never scored
against silver labels, because silver labels ARE the baseline's output (that score would be 100%
and meaningless).
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.baseline.dictionary_ner import DictionaryNER
from src.evaluation.metrics import evaluate_entities, format_report
from src.labeling.bio import LABELS
from src.utils.config import project_path
from src.utils.io import read_jsonl
from src.utils.validation import validate_records

BLUE, INK, MUTED = "#2a78d6", "#0b0b0b", "#52514e"


def load_gold_or_exit(split: str) -> list:
    path = project_path(f"data/splits/gold/{split}.jsonl")
    if not path.exists():
        raise SystemExit(f"{path.relative_to(project_path(''))} does not exist yet.\n"
                         "Annotate the gold batches and run `python -m src.annotation.import_annotations` first.")
    records = read_jsonl(path)
    validate_records(records, name=f"gold {split}")
    if any(r.get("status") != "verified" or r.get("label_source") != "gold_human" for r in records):
        raise SystemExit("Gold file contains records that are not human-verified - refusing to evaluate.")
    return records


def predict_all(records: list) -> list:
    ner = DictionaryNER()
    return [ner.predict_record(r) for r in records]


def label_confusion(gold_records, predictions) -> pd.DataFrame:
    """Rows: gold label (+ NONE for spurious predictions); columns: predicted label (+ MISSED).
    Entities are paired when their spans overlap."""
    rows = []
    for record, pred in zip(gold_records, predictions):
        used = set()
        for g in record["entities"]:
            overlap = [i for i, p in enumerate(pred) if p["start"] < g["end"] and g["start"] < p["end"]]
            if overlap:
                used.update(overlap)
                rows.append((g["label"], pred[overlap[0]]["label"]))
            else:
                rows.append((g["label"], "MISSED"))
        rows += [("NONE", p["label"]) for i, p in enumerate(pred) if i not in used]
    table = pd.crosstab(pd.Series([r[0] for r in rows], name="gold"), pd.Series([r[1] for r in rows], name="predicted"))
    return table.reindex(index=LABELS + ["NONE"], columns=LABELS + ["MISSED"], fill_value=0)


def plot_results(result, confusion, split):
    per_class = pd.DataFrame(result["per_class"]).T
    per_class = per_class[per_class["support"] > 0].sort_values("f1")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(per_class.index, per_class["f1"], color=BLUE, height=0.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"entity-level F1 (gold {split} set)")
    ax.set_title("Dictionary/rule baseline: F1 per class", loc="left", color=INK)
    for i, (f1, n) in enumerate(zip(per_class["f1"], per_class["support"])):
        ax.text(f1 + 0.01, i, f"{f1:.2f}  (n={int(n)})", va="center", fontsize=8, color=MUTED)
    fig.savefig(project_path("reports/figures/09_baseline_per_class_f1.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(confusion.values, cmap="Blues")
    ax.set_xticks(range(len(confusion.columns)), confusion.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(confusion.index)), confusion.index, fontsize=8)
    ax.set_xlabel("predicted label")
    ax.set_ylabel("gold label")
    vmax = confusion.values.max() or 1
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            v = confusion.values[i, j]
            if v:
                ax.text(j, i, v, ha="center", va="center", fontsize=7, color="white" if v > vmax / 2 else INK)
    ax.set_title("Which gold labels become which predicted labels (overlapping spans)", loc="left", color=INK)
    fig.savefig(project_path("reports/figures/10_baseline_label_confusion.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    args = parser.parse_args()

    gold = load_gold_or_exit(args.split)
    predictions = predict_all(gold)
    result = evaluate_entities([r["entities"] for r in gold], predictions)
    confusion = label_confusion(gold, predictions)

    project_path("reports/baseline_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    confusion.to_csv(project_path("reports/baseline_label_confusion.csv"))
    markdown = format_report(result, f"Dictionary/rule baseline on gold {args.split} ({len(gold)} products)")
    project_path("reports/baseline_metrics.md").write_text(markdown + "\n", encoding="utf-8")
    plot_results(result, confusion, args.split)
    print(markdown)


if __name__ == "__main__":
    main()
