"""Compare the trained recognisers: one table and two figures from reports/ocr_model/<run>.json.

    python -m src.ocr_model.compare                       # runs ann, cnn, crnn
    python -m src.ocr_model.compare --runs ann crnn

Writes reports/ocr_model/comparison.md, reports/figures/ocr_model/test_cer.png and validation_curves.png.
"""
import argparse
import json

from src.utils.config import project_path

LABELS = {"ann": "ANN (MLP)", "cnn": "CNN", "crnn": "CRNN (CNN + BiLSTM)"}
SEVERITY = {"severity_0": "clean print", "severity_1": "phone photo", "severity_2": "hard photo"}


def load(runs: list) -> dict:
    found = {}
    for run in runs:
        path = project_path(f"reports/ocr_model/{run}.json")
        if path.exists():
            found[run] = json.loads(path.read_text())
    return found


def table(results: dict) -> str:
    head = ("| Model | Parameters | Steps | Train min | " + " | ".join(f"CER {v}" for v in SEVERITY.values())
            + " | WER phone photo | Lines exact (phone) | ms / line |")
    rows = [head, "|" + "---|" * (head.count("|") - 1)]
    for run, r in results.items():
        t = r["test"]
        rows.append(f"| {LABELS.get(r['model'], run)} | {r['parameters']:,} | {r['steps_done']:,} | {r['train_minutes']} | "
                    + " | ".join(f"{t[s]['cer']:.3f}" for s in SEVERITY)
                    + f" | {t['severity_1']['wer']:.3f} | {t['severity_1']['line_accuracy']:.1%}"
                    + f" | {t['severity_1']['ms_per_line']:.1f} |")
    return "\n".join(rows)


def figures(results: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    out = project_path("reports/figures/ocr_model")
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    width = 0.8 / max(len(results), 1)
    for i, (run, r) in enumerate(results.items()):
        values = [r["test"][s]["cer"] for s in SEVERITY]
        bars = ax.bar(np.arange(3) + i * width, values, width, label=LABELS.get(r["model"], run))
        ax.bar_label(bars, fmt="%.3f", fontsize=7)
    ax.set_xticks(np.arange(3) + width * (len(results) - 1) / 2, list(SEVERITY.values()))
    ax.set_ylabel("character error rate (lower is better)")
    ax.set_title("Test lines: unseen products, unseen fonts")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "test_cer.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    for run, r in results.items():
        steps = [h["step"] for h in r["history"]]
        ax.plot(steps, [h["val_cer"] for h in r["history"]], marker="o", markersize=3, label=LABELS.get(r["model"], run))
    ax.set_xlabel("training step")
    ax.set_ylabel("validation CER")
    ax.set_ylim(0, 1)
    ax.set_title("Learning curves")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "validation_curves.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["ann", "cnn", "crnn"])
    results = load(parser.parse_args().runs)
    if not results:
        raise SystemExit("no results yet: train first, e.g. python -m src.ocr_model.train --model crnn")
    report = ["# Text-line recognisers: ANN vs CNN vs CRNN", "",
              "Same synthetic training data, CTC loss, optimiser and schedule for every model. Test lines come from "
              "test-split products, rendered in fonts never used for training. CER = character edits / reference "
              "characters (greedy CTC decoding).", "", table(results), ""]
    for run, r in results.items():
        report += [f"## {LABELS.get(r['model'], run)}: example readings (phone photo)", "",
                   "| Reference | Prediction |", "|---|---|"]
        report += [f"| `{e['reference']}` | `{e['prediction']}` |" for e in r["test"]["severity_1"]["examples"][:6]]
        report.append("")
    path = project_path("reports/ocr_model/comparison.md")
    path.write_text("\n".join(report), encoding="utf-8")
    figures(results)
    print(table(results))
    print("saved", path)


if __name__ == "__main__":
    main()
