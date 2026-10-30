"""Optimiser and hyper-parameter study for our CRNN (resumable: finished runs are skipped).

    python -m src.ocr_model.optimizer_study

1. Sweep: 6 optimisers x 3 learning rates, each trained for STEPS steps on the tough synthetic lines with the
   same data order, seed, batch size 32, one-cycle schedule (momentum not cycled, so the optimisers differ only
   in their update rule) and gradient clipping:
     sgd (plain mini-batch SGD), momentum (SGD + 0.9), nesterov (SGD + Nesterov 0.9), rmsprop, adam, adamw
2. Batch size: the best setting again with batch 16 and 64 (same number of steps).
3. Selection uses VALIDATION data only: the Open Food Facts real crops (train/validation-split products, none of
   the 60 test photos), with the synthetic validation CER reported alongside. The 377 real test crops are not
   looked at until the final step.
4. Final: the winner is trained for the full 12,000 steps (run "crnn_tuned"), then put through the same
   real-photo pipeline as before (Open Food Facts + test-half fine-tuning, two folds, beam search with the food
   LM) via improve_crnn.run(base="crnn_tuned"), and compared with the current 9.18%.
Writes reports/ocr_model/optimizer_study.json/.md and figures in reports/figures/ocr_model/.
"""
import json

import torch

from src.ocr_model.evaluate import error_rates
from src.ocr_model.finetune import crnn_read
from src.ocr_model.improve_crnn import off_items
from src.ocr_model.real_photos import load_model
from src.ocr_model.train import parse_args, train
from src.utils.config import project_path

STEPS = 2500
GRID = {"sgd": (0.01, 0.03, 0.1), "momentum": (0.003, 0.01, 0.03), "nesterov": (0.003, 0.01, 0.03),
        "rmsprop": (1e-4, 3e-4, 1e-3), "adam": (3e-4, 1e-3, 3e-3), "adamw": (3e-4, 1e-3, 3e-3)}
OUT = project_path("reports/ocr_model/optimizer_study.json")


def run_one(optimizer: str, lr: float, batch_size: int = 32, steps: int = STEPS, name: str = None) -> dict:
    name = name or f"optimizer_study/{optimizer}_lr{lr:g}_bs{batch_size}"
    report = project_path(f"reports/ocr_model/{name}.json")
    if not report.exists():
        args = parse_args(["--model", "crnn", "--tough", "--no-cycle-momentum", "--optimizer", optimizer,
                           "--lr", str(lr), "--batch-size", str(batch_size), "--steps", str(steps),
                           "--eval-every", "250", "--patience", "99", "--val-size", "500", "--test-size", "300",
                           "--run", name])
        train(args)
    result = json.loads(report.read_text())
    return {"run": name, "optimizer": optimizer, "lr": lr, "batch_size": batch_size, "steps": steps,
            "synthetic_val_cer": result["best"]["val_cer"], "best_step": result["best"]["step"],
            "train_minutes": result["train_minutes"],
            "curve": [(h["step"], h["val_cer"], h["train_loss"]) for h in result["history"]]}


def real_val_cer(run_name: str, items: list, device) -> float:
    model = load_model(run_name, device)
    preds = crnn_read(model, [img for img, _ in items], device)
    return error_rates(preds, [label for _, label in items])["cer"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = off_items()
    study = json.loads(OUT.read_text()) if OUT.exists() else {"runs": []}
    done = {r["run"] for r in study["runs"]}

    def record(row):
        row["real_val_cer"] = real_val_cer(row["run"], items, device)
        study["runs"].append(row)
        OUT.write_text(json.dumps(study, indent=2))
        print(f"{row['run']}: synthetic val CER {row['synthetic_val_cer']:.4f}, real val CER {row['real_val_cer']:.4f}",
              flush=True)

    for optimizer, lrs in GRID.items():
        for lr in lrs:
            name = f"optimizer_study/{optimizer}_lr{lr:g}_bs32"
            if name not in done:
                record(run_one(optimizer, lr))

    sweep = [r for r in study["runs"] if r["batch_size"] == 32]
    best = min(sweep, key=lambda r: (r["real_val_cer"], r["synthetic_val_cer"]))
    for bs in (16, 64):
        name = f"optimizer_study/{best['optimizer']}_lr{best['lr']:g}_bs{bs}"
        if name not in {r["run"] for r in study["runs"]}:
            record(run_one(best["optimizer"], best["lr"], bs))
    candidates = [r for r in study["runs"] if r["optimizer"] == best["optimizer"] and r["lr"] == best["lr"]]
    winner = min(candidates, key=lambda r: (r["real_val_cer"], r["synthetic_val_cer"]))
    study["winner"] = {k: winner[k] for k in ("optimizer", "lr", "batch_size", "real_val_cer", "synthetic_val_cer")}
    OUT.write_text(json.dumps(study, indent=2))
    print("winner:", study["winner"], flush=True)

    # full training of the winner, then the same real-photo pipeline as the current best CRNN
    if not project_path("reports/ocr_model/crnn_tuned.json").exists():
        train(parse_args(["--model", "crnn", "--tough", "--no-cycle-momentum", "--optimizer", winner["optimizer"],
                          "--lr", str(winner["lr"]), "--batch-size", str(winner["batch_size"]), "--run", "crnn_tuned"]))
    from src.ocr_model.improve_crnn import run as improve
    final = improve(base="crnn_tuned")
    current = json.loads(project_path("reports/ocr_model/improve_crnn.json").read_text())["systems"]
    key = "+ Open Food Facts crops + beam search with food LM"
    study["final"] = {"tuned": final["systems"][key], "current": current[key],
                      "tuned_greedy": final["systems"]["+ Open Food Facts crops"],
                      "current_greedy": current["+ Open Food Facts crops"],
                      "rapidocr": current["RapidOCR (reference)"]}
    OUT.write_text(json.dumps(study, indent=2))
    report(study)
    print("final (377 real test crops): tuned CER", round(study["final"]["tuned"]["cer"], 4), "| current",
          round(study["final"]["current"]["cer"], 4), "| RapidOCR", round(study["final"]["rapidocr"]["cer"], 4))


NAMES = {"sgd": "SGD", "momentum": "SGD + momentum", "nesterov": "SGD + Nesterov", "rmsprop": "RMSprop",
         "adam": "Adam", "adamw": "AdamW"}


def report(study: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sweep = [r for r in study["runs"] if r["batch_size"] == 32]
    figs = project_path("reports/figures/ocr_model")

    # learning curves: best learning rate of each optimiser
    fig, ax = plt.subplots(figsize=(8, 4))
    for opt in GRID:
        runs = [r for r in sweep if r["optimizer"] == opt]
        if not runs:
            continue
        b = min(runs, key=lambda r: r["real_val_cer"])
        steps, cers = zip(*[(s, c) for s, c, _ in b["curve"]])
        ax.plot(steps, cers, marker="o", markersize=2.5, label=f"{NAMES[opt]} (lr {b['lr']:g})")
    ax.set_ylim(0, 1)
    ax.set_xlabel("training step")
    ax.set_ylabel("synthetic validation CER")
    ax.set_title("Learning curves, best learning rate per optimiser", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figs / "optimizer_curves.png", dpi=160)
    plt.close(fig)

    # optimiser x learning rate: real validation CER
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for i, opt in enumerate(GRID):
        runs = sorted([r for r in sweep if r["optimizer"] == opt], key=lambda r: r["lr"])
        for j, r in enumerate(runs):
            x = i + (j - 1) * 0.27
            bar = ax.bar(x, min(r["real_val_cer"], 1.0), 0.25, color=["#C9DDD7", "#5FA898", "#0E5C4C"][j])
            ax.text(x, min(r["real_val_cer"], 1.0) + 0.01, f"{r['lr']:g}", ha="center", fontsize=6.5, rotation=90)
    ax.set_xticks(range(len(GRID)), [NAMES[o] for o in GRID], fontsize=8)
    ax.set_ylabel("CER on real validation crops")
    ax.set_title("Optimiser x learning rate (label = learning rate; lower is better)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figs / "optimizer_grid.png", dpi=160)
    plt.close(fig)

    w = study["winner"]
    lines = ["# Optimiser and hyper-parameter study (CRNN)", "",
             f"Each run: {STEPS} steps, tough synthetic lines, seed 42, one-cycle schedule without momentum cycling, "
             "gradient clipping 5. Selection on real VALIDATION crops (Open Food Facts, no test photos).", "",
             "| Optimiser | Learning rate | Batch | Synthetic val CER | Real val CER | Minutes |", "|---|---|---|---|---|---|"]
    for r in sorted(study["runs"], key=lambda r: (list(GRID).index(r["optimizer"]), r["batch_size"], r["lr"])):
        lines.append(f"| {NAMES[r['optimizer']]} | {r['lr']:g} | {r['batch_size']} | {r['synthetic_val_cer']:.4f} | "
                     f"{r['real_val_cer']:.4f} | {r['train_minutes']} |")
    lines += ["", f"**Winner:** {NAMES[w['optimizer']]}, learning rate {w['lr']:g}, batch {w['batch_size']} "
              f"(real val CER {w['real_val_cer']:.4f})."]
    if "final" in study:
        f = study["final"]
        lines += ["", "## Final test: 377 real crops (two-fold by photo, after the full real-photo pipeline)", "",
                  "| CRNN | Greedy CER | + beam search CER | + beam, ignoring case |", "|---|---|---|---|",
                  f"| Current (AdamW 1e-3, momentum cycled) | {f['current_greedy']['cer']:.4f} | {f['current']['cer']:.4f} | "
                  f"{f['current']['cer_ignoring_case']:.4f} |",
                  f"| Tuned ({NAMES[w['optimizer']]} {w['lr']:g}, batch {w['batch_size']}) | {f['tuned_greedy']['cer']:.4f} | "
                  f"{f['tuned']['cer']:.4f} | {f['tuned']['cer_ignoring_case']:.4f} |",
                  f"| RapidOCR (reference) | {f['rapidocr']['cer']:.4f} | | {f['rapidocr']['cer_ignoring_case']:.4f} |"]
    project_path("reports/ocr_model/optimizer_study.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
