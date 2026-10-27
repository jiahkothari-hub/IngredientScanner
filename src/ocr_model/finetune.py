"""Domain adaptation: fine-tune the synthetic-trained CRNN on real line crops, tested on photos it never saw.

    python -m src.ocr_model.finetune                      # from models/runs/ocr/crnn_tough
    python -m src.ocr_model.finetune --base crnn --steps 1500

Two-fold split BY PHOTO (crops of one photo are never in both halves):
  fold 0: fine-tune on the crops of photo half A, test on half B;  fold 1: the other way round.
Every crop is therefore read by a fine-tuned model that has never seen its photo, and the two folds together
cover all real crops. Training batches mix real crops (light augmentation: contrast, blur, width, noise) with
synthetic lines, so the model adapts to real print without forgetting what it learned from 384k synthetic
lines. A fixed number of steps is used: nothing is chosen by looking at the test half.

Compared on the same crops: RapidOCR's recogniser, the first CRNN, the CRNN trained on tougher synthetic lines,
and that CRNN after fine-tuning. Writes reports/ocr_model/finetune.json/.md and a figure.
"""
import argparse
import json
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.ocr_model.evaluate import error_rates
from src.ocr_model.real_photos import CROP_DIR, load_model, pad_width, strip_header
from src.ocr_model.real_photos import crnn_read as _crnn_read
from src.ocr_model.render_lines import HEIGHT, SyntheticLines, collate
from src.utils.config import project_path


def crnn_read(model, images, device) -> list:
    """CRNN readings with the "Ingredients:" heading removed, exactly as for the stored test-set readings."""
    return [strip_header(t) for t in _crnn_read(model, images, device)]


def load_rows() -> pd.DataFrame:
    return pd.read_csv(CROP_DIR / "labels.csv", dtype=str, keep_default_na=False)


def photo_halves(rows: pd.DataFrame, seed: int = 0) -> tuple:
    photos = sorted(rows["product_id"].unique())
    random.Random(seed).shuffle(photos)
    half = len(photos) // 2
    return set(photos[:half]), set(photos[half:])


def read_crop(name: str) -> np.ndarray:
    return cv2.imread(str(CROP_DIR / name), cv2.IMREAD_GRAYSCALE)


def augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Light photometric/width jitter for real crops (the text itself is never changed)."""
    img = img.astype(np.float32)
    img = img * rng.uniform(0.7, 1.3) + rng.uniform(-30, 30)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.0))
    w = max(8, round(img.shape[1] * rng.uniform(0.85, 1.15)))
    img = cv2.resize(img, (w, HEIGHT), interpolation=cv2.INTER_LINEAR)
    img += np.random.default_rng(rng.randrange(2**31)).normal(0, rng.uniform(0, 6), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def real_item(img: np.ndarray, label: str):
    img = pad_width(img, len(label) + sum(a == b for a, b in zip(label, label[1:])))
    return torch.from_numpy(img).float().div(255).unsqueeze(0), label


def finetune(base: str, train_rows: pd.DataFrame, steps: int, lr: float, real_per_batch: int, synth_per_batch: int,
             seed: int, device, extra: list = ()) -> nn.Module:
    """extra: further real (grey 32-px image, label) pairs, e.g. Open Food Facts crops (improve_crnn.py)."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = load_model(base, device)
    images = {r.crop: read_crop(r.crop) for r in train_rows.itertuples()}
    items = list(zip(train_rows["crop"], train_rows["label"]))
    for i, (img, label) in enumerate(extra):
        images[f"extra_{i}"] = img
        items.append((f"extra_{i}", label))
    synthetic = SyntheticLines("train", 10**9, seed=1000 + seed, severity=1, tough=True)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=steps)
    model.train()
    for step in range(steps):
        batch = [real_item(augment(images[c], rng), t) for c, t in (rng.choice(items) for _ in range(real_per_batch))]
        batch += [synthetic[step * synth_per_batch + i] for i in range(synth_per_batch)]
        b = collate(batch)
        loss = ctc(model(b["images"].to(device)), b["targets"].to(device), b["input_lengths"].to(device),
                   b["target_lengths"].to(device))
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
        schedule.step()
        if (step + 1) % 250 == 0:
            print(f"  step {step + 1}  loss {loss.item():.4f}", flush=True)
    return model.eval()


def run(base: str = "crnn_tough", steps: int = 1500, lr: float = 3e-4, real_per_batch: int = 16,
        synth_per_batch: int = 16, seed: int = 0) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_rows()
    halves = photo_halves(rows, seed)
    base_model = load_model(base, device)
    readings = {"rapidocr": dict(zip(rows["crop"], rows["rapidocr"])), "crnn": dict(zip(rows["crop"], rows["crnn"])),
                base: {}, f"{base}_finetuned": {}}
    readings[base] = dict(zip(rows["crop"], crnn_read(base_model, [read_crop(c) for c in rows["crop"]], device)))
    folds = []
    for k, (train_photos, test_photos) in enumerate([(halves[0], halves[1]), (halves[1], halves[0])]):
        train_rows = rows[rows["product_id"].isin(train_photos)]
        test_rows = rows[rows["product_id"].isin(test_photos)]
        print(f"fold {k}: fine-tune on {len(train_rows)} crops ({len(train_photos)} photos), "
              f"test on {len(test_rows)} crops ({len(test_photos)} photos)", flush=True)
        model = finetune(base, train_rows, steps, lr, real_per_batch, synth_per_batch, seed + k, device)
        out = project_path(f"models/runs/ocr/{base}_finetuned_fold{k}")
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"model": "crnn", "state_dict": model.state_dict(), "base": base, "fold": k,
                    "train_photos": sorted(train_photos)}, out / "best.pt")
        preds = crnn_read(model, [read_crop(c) for c in test_rows["crop"]], device)
        readings[f"{base}_finetuned"].update(zip(test_rows["crop"], preds))
        folds.append({"fold": k, "train_crops": len(train_rows), "test_crops": len(test_rows),
                      "test_cer": error_rates(preds, test_rows["label"].tolist())["cer"]})

    labels = rows["label"].tolist()
    systems = {}
    for name, reading in readings.items():
        preds = [reading[c] for c in rows["crop"]]
        exact = error_rates(preds, labels)
        nocase = error_rates([p.lower() for p in preds], [l.lower() for l in labels])
        systems[name] = {"cer": exact["cer"], "wer": exact["wer"], "line_accuracy": exact["line_accuracy"],
                         "cer_ignoring_case": nocase["cer"], "wer_ignoring_case": nocase["wer"]}
    per_photo = {}
    for pid, g in rows.groupby("product_id"):
        per_photo[pid] = {n: error_rates([readings[n][c] for c in g["crop"]], g["label"].tolist())["cer"]
                          for n in readings}
    ft = f"{base}_finetuned"
    result = {"base": base, "crops": len(rows), "photos": rows["product_id"].nunique(),
              "settings": {"steps": steps, "lr": lr, "real_per_batch": real_per_batch,
                           "synthetic_per_batch": synth_per_batch, "seed": seed},
              "folds": folds, "systems": systems,
              "photos_finetuned_beats_rapidocr": sum(v[ft] < v["rapidocr"] for v in per_photo.values()),
              "photos_rapidocr_beats_finetuned": sum(v["rapidocr"] < v[ft] for v in per_photo.values()),
              "per_photo_cer": per_photo,
              "examples": [{"label": l, "rapidocr": readings["rapidocr"][c], "before": readings[base][c],
                            "after": readings[ft][c]}
                           for c, l in zip(rows["crop"], labels) if readings[base][c] != readings[ft][c]][:25]}
    project_path("reports/ocr_model/finetune.json").write_text(json.dumps(result, indent=2))
    write_report(result)
    return result


NAMES = {"rapidocr": "RapidOCR recogniser (pretrained, PP-OCRv6)", "crnn": "CRNN v1 (synthetic)",
         "crnn_tough": "CRNN v2 (tougher synthetic)", "crnn_tough_finetuned": "CRNN v2 + fine-tuned on real crops"}


def write_report(result: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lines = ["# Closing the synthetic-to-real gap", "",
             f"{result['crops']} real line crops from {result['photos']} packet photos, two-fold split by photo: "
             "every fine-tuned reading comes from a model that never saw that photo.", "",
             "| Recogniser | CER | WER | CER (ignoring case) | Lines exact |", "|---|---|---|---|---|"]
    for name, m in result["systems"].items():
        lines.append(f"| {NAMES.get(name, name)} | {m['cer']:.3f} | {m['wer']:.3f} | {m['cer_ignoring_case']:.3f} | "
                     f"{m['line_accuracy']:.1%} |")
    lines += ["", f"Per photo: fine-tuned CRNN better than RapidOCR on {result['photos_finetuned_beats_rapidocr']} "
              f"photos, RapidOCR better on {result['photos_rapidocr_beats_finetuned']}.", "",
              "Folds: " + "; ".join(f"fold {f['fold']} trained on {f['train_crops']} crops, test CER "
                                    f"{f['test_cer']:.3f} on {f['test_crops']}" for f in result["folds"]), "",
              "## Readings changed by fine-tuning", "", "| Label | Before | After | RapidOCR |", "|---|---|---|---|"]
    lines += [f"| `{e['label']}` | `{e['before']}` | `{e['after']}` | `{e['rapidocr']}` |" for e in result["examples"]]
    project_path("reports/ocr_model/finetune.md").write_text("\n".join(lines), encoding="utf-8")

    names = list(result["systems"])
    values = [result["systems"][n]["cer"] for n in names]
    colours = ["#9FB9B2" if n == "rapidocr" else "#0E7C66" if n.endswith("finetuned") else "#C9B8A6" for n in names]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    bars = ax.barh([NAMES.get(n, n) for n in names][::-1], values[::-1], color=colours[::-1])
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xlabel("character error rate on real packet crops (lower is better)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(project_path("reports/figures/ocr_model/real_photo_cer.png"), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="crnn_tough")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    a = parser.parse_args()
    result = run(a.base, a.steps, a.lr, seed=a.seed)
    for name, m in result["systems"].items():
        print(f"{NAMES.get(name, name):45s} CER {m['cer']:.3f}  WER {m['wer']:.3f}  exact {m['line_accuracy']:.1%}")


if __name__ == "__main__":
    main()
