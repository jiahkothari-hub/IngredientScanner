"""Our fine-tuned CRNN vs RapidOCR's recogniser on the real packet crops: numbers and figures.

    python -m src.ocr_model.compare_real        (after python -m src.ocr_model.finetune)

Every crop is read by the fine-tuned fold model that did NOT see its photo (same split as finetune.py).
Writes reports/ocr_model/crnn_vs_rapidocr.md and reports/figures/ocr_model/crnn_vs_rapidocr.png.
"""
import json

import numpy as np
import pandas as pd
import torch
from rapidfuzz.distance import Levenshtein

from src.ocr_model.evaluate import error_rates
from src.ocr_model.finetune import crnn_read, load_rows, photo_halves, read_crop
from src.ocr_model.real_photos import load_model
from src.utils.config import project_path

OURS, THEIRS = "#0E7C66", "#9FB9B2"


def crop_readings(base: str = "crnn_tough") -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_rows()
    halves = photo_halves(rows, 0)
    rows["crnn_ft"] = ""
    for k, test_photos in enumerate([halves[1], halves[0]]):          # fold k was trained on the other half
        model = load_model(f"{base}_finetuned_fold{k}", device)
        mask = rows["product_id"].isin(test_photos)
        rows.loc[mask, "crnn_ft"] = crnn_read(model, [read_crop(c) for c in rows.loc[mask, "crop"]], device)
    for s in ("rapidocr", "crnn_ft"):
        rows[f"{s}_cer"] = [Levenshtein.distance(p, l) / max(len(l), 1) for p, l in zip(rows[s], rows["label"])]
    return rows


def char_confusions(rows: pd.DataFrame, system: str, top: int = 8) -> list:
    """Most frequent single-character substitutions (reference -> reading)."""
    counts = {}
    for pred, label in zip(rows[system], rows["label"]):
        for op in Levenshtein.editops(label, pred):
            if op.tag == "replace":
                key = (label[op.src_pos], pred[op.dest_pos])
                counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])[:top]


def figure(rows: pd.DataFrame, summary: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))

    # 1. headline metrics
    ax = axes[0, 0]
    metrics = [("CER", "cer"), ("CER\n(ignoring case)", "cer_ignoring_case"), ("WER", "wer"),
               ("lines not\nexactly right", "lines_wrong")]
    x = np.arange(len(metrics))
    for i, (name, colour, label) in enumerate([("rapidocr", THEIRS, "RapidOCR (pretrained)"),
                                               ("crnn_ft", OURS, "Our CRNN v2 + fine-tuning")]):
        vals = [summary[name][m] for _, m in metrics]
        bars = ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, color=colour, label=label)
        ax.bar_label(bars, fmt="%.3f", fontsize=7.5, padding=2)
    ax.set_xticks(x, [m for m, _ in metrics], fontsize=8.5)
    ax.set_ylabel("error rate (lower is better)")
    ax.set_title("Error rates on 377 real crops (57 photos)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    # 2. per photo
    ax = axes[0, 1]
    per_photo = rows.groupby("product_id").apply(
        lambda g: pd.Series({s: error_rates(g[s].tolist(), g["label"].tolist())["cer"] for s in ("rapidocr", "crnn_ft")}),
        include_groups=False)
    ax.scatter(per_photo["rapidocr"], per_photo["crnn_ft"], s=22, color=OURS, alpha=0.75)
    lim = max(per_photo.max().max(), 0.05) * 1.05
    ax.plot([0, lim], [0, lim], color="#888", lw=1, ls="--")
    ax.text(lim * 0.97, lim * 0.06, "below the line:\nour CRNN better", ha="right", fontsize=8, color="#555")
    ax.text(lim * 0.03, lim * 0.9, "above the line:\nRapidOCR better", ha="left", fontsize=8, color="#555")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("RapidOCR CER per photo")
    ax.set_ylabel("our CRNN CER per photo")
    ax.set_title(f"Per photo: ours better on {int((per_photo['crnn_ft'] < per_photo['rapidocr']).sum())}, "
                 f"RapidOCR better on {int((per_photo['rapidocr'] < per_photo['crnn_ft']).sum())}, "
                 f"tied on {int((per_photo['rapidocr'] == per_photo['crnn_ft']).sum())}", fontsize=10)

    # 3. distribution of per-crop errors
    ax = axes[1, 0]
    bins = [0, 1e-9, 0.05, 0.10, 0.25, 0.5, 10]
    names = ["perfect", "≤5%", "5–10%", "10–25%", "25–50%", ">50%"]
    x = np.arange(len(names))
    for i, (s, colour, label) in enumerate([("rapidocr", THEIRS, "RapidOCR"), ("crnn_ft", OURS, "Our CRNN")]):
        counts = pd.cut(rows[f"{s}_cer"], bins=bins, labels=names, include_lowest=True).value_counts()[names]
        bars = ax.bar(x + (i - 0.5) * 0.38, counts.values / len(rows), 0.38, color=colour, label=label)
        ax.bar_label(bars, labels=[f"{v:.0%}" for v in counts.values / len(rows)], fontsize=7, padding=2)
    ax.set_xticks(x, names, fontsize=8.5)
    ax.set_ylabel("share of crops")
    ax.set_xlabel("character error rate of the crop")
    ax.set_title("How wrong is each line?", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    # 4. by region
    ax = axes[1, 1]
    groups = rows.groupby("country_group")
    regions = sorted(groups.groups)
    x = np.arange(len(regions))
    for i, (s, colour, label) in enumerate([("rapidocr", THEIRS, "RapidOCR"), ("crnn_ft", OURS, "Our CRNN")]):
        vals = [error_rates(groups.get_group(r)[s].tolist(), groups.get_group(r)["label"].tolist())["cer"] for r in regions]
        bars = ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, color=colour, label=label)
        ax.bar_label(bars, fmt="%.3f", fontsize=7.5, padding=2)
    ax.set_xticks(x, [f"{r.replace('_', ' ')}\n({len(groups.get_group(r))} crops)" for r in regions], fontsize=8)
    ax.set_ylabel("CER")
    ax.set_title("By market", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Our fine-tuned CRNN vs RapidOCR's recogniser, same real line crops", fontsize=12)
    fig.tight_layout()
    path = project_path("reports/figures/ocr_model/crnn_vs_rapidocr.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, per_photo


def main():
    rows = crop_readings()
    summary = {}
    for s in ("rapidocr", "crnn_ft"):
        exact = error_rates(rows[s].tolist(), rows["label"].tolist())
        nocase = error_rates(rows[s].str.lower().tolist(), rows["label"].str.lower().tolist())
        summary[s] = {"cer": exact["cer"], "wer": exact["wer"], "cer_ignoring_case": nocase["cer"],
                      "lines_wrong": 1 - exact["line_accuracy"], "line_accuracy": exact["line_accuracy"],
                      "perfect_crops": int((rows[f"{s}_cer"] == 0).sum()),
                      "confusions": [f"{a!r}->{b!r} x{n}" for (a, b), n in char_confusions(rows, s)]}
    path, per_photo = figure(rows, summary)
    both = rows[(rows["crnn_ft"] == rows["label"]) & (rows["rapidocr"] != rows["label"])]
    only_rapid = rows[(rows["rapidocr"] == rows["label"]) & (rows["crnn_ft"] != rows["label"])]
    lines = ["# Our fine-tuned CRNN vs RapidOCR (same real crops)", "",
             "| | RapidOCR (pretrained) | Our CRNN v2 + fine-tuning |", "|---|---|---|"]
    for key, name in [("cer", "Character error rate"), ("cer_ignoring_case", "CER ignoring case"),
                      ("wer", "Word error rate"), ("line_accuracy", "Lines read exactly"),
                      ("perfect_crops", "Perfect crops")]:
        fmt = (lambda v: f"{v:.1%}") if key == "line_accuracy" else (lambda v: str(v)) if key == "perfect_crops" \
            else (lambda v: f"{v:.3f}")
        lines.append(f"| {name} | {fmt(summary['rapidocr'][key])} | {fmt(summary['crnn_ft'][key])} |")
    lines += ["", f"Crops only our CRNN read exactly: {len(both)}; only RapidOCR read exactly: {len(only_rapid)}.", "",
              "Most frequent character substitutions:", "",
              f"- RapidOCR: {', '.join(summary['rapidocr']['confusions'])}",
              f"- Our CRNN: {', '.join(summary['crnn_ft']['confusions'])}", "",
              "## Crops our CRNN read exactly and RapidOCR did not", "", "| Label | RapidOCR |", "|---|---|"]
    lines += [f"| `{r.label}` | `{r.rapidocr}` |" for r in both.head(10).itertuples()]
    lines += ["", "## Worst crops for our CRNN", "", "| Label | Ours | RapidOCR |", "|---|---|---|"]
    lines += [f"| `{r.label}` | `{r.crnn_ft}` | `{r.rapidocr}` |"
              for r in rows.sort_values("crnn_ft_cer", ascending=False).head(10).itertuples()]
    project_path("reports/ocr_model/crnn_vs_rapidocr.md").write_text("\n".join(lines), encoding="utf-8")
    project_path("reports/ocr_model/crnn_vs_rapidocr.json").write_text(json.dumps(
        {"summary": summary, "only_crnn_exact": len(both), "only_rapidocr_exact": len(only_rapid),
         "per_photo_cer": per_photo.to_dict(orient="index")}, indent=2))
    print("\n".join(lines[:9]))
    print(lines[10])
    print("figure:", path)


if __name__ == "__main__":
    main()
