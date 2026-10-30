"""Transfer learning: fine-tune RapidOCR's own pretrained recogniser (PP-OCRv6 small) on food-label crops.

    python -m src.ocr_model.ppocr_finetune                 # verify, fine-tune both folds, compare

The recogniser RapidOCR runs (PP-OCRv6_rec_small, 5.3M parameters, trained by Baidu/PaddleOCR on large real
data) is loaded in PyTorch from RapidOCR's published weights (models/pretrained/ppocrv6_rec_small, SHA-256
checked) using the network code that ships with RapidOCR. Steps:
  1. Re-cut the real test crops (data/ocr_crops/real/labels.csv) from the photos in colour at full resolution,
     because this model reads 48-pixel-high colour lines (our CRNN crops are 32-pixel grey).
  2. Check the PyTorch copy reproduces RapidOCR: same preprocessing (height 48, minimum width ratio 320/48,
     pixels scaled to [-1, 1], zero padding on the right) and the same CTC decoding.
  3. Two-fold split by photo (same halves as finetune.py). Inside each TRAINING half only:
       - crops whose label does not line up with the crop at the start or end of the line (RapidOCR's reading
         has 2+ extra or missing characters there) are dropped; misreadings inside a line are kept;
       - a quarter of its photos are held out for validation, and the learning rate (1e-5 / 3e-5 / 1e-4) and
         number of steps (every 50 up to 400) with the lowest validation CER are chosen, including "0 steps"
         (keep the pretrained model) as an option;
       - the model is then fine-tuned on the whole cleaned training half with that setting.
     The test half is never filtered and never used for any choice. Only the CTC branch is trained
     (PaddleOCR's auxiliary training heads are not part of the released model); batch-norm statistics stay
     frozen because there are only ~190 crops per half.
Writes reports/ocr_model/ppocr_finetune.json/.md and reports/figures/ocr_model/ocr_all_models.png.
"""
import argparse
import json
import math
import random

import cv2
import numpy as np
import pandas as pd
import torch
from rapidfuzz.distance import Levenshtein
from torch import nn

from src.ocr_model.charset import clean_text
from src.ocr_model.evaluate import error_rates
from src.ocr_model.finetune import load_rows, photo_halves
from src.ocr_model.real_photos import detect_and_read, strip_header
from src.utils.config import project_path

WEIGHTS_DIR = project_path("models/pretrained/ppocrv6_rec_small")
COLOR_DIR = project_path("data/ocr_crops/real_color")
HEIGHT, MIN_RATIO = 48, 320 / 48


# ---------------------------------------------------------------------------------------- model
def load_ppocr(device, state_dict=None):
    """(model, characters): PP-OCRv6_rec_small in PyTorch with RapidOCR's released weights."""
    from omegaconf import OmegaConf
    from rapidocr.inference_engine.pytorch.networks.architectures.base_model import BaseModel
    from rapidocr.inference_engine.pytorch.networks.main import ModelLoader
    model = BaseModel(OmegaConf.load(ModelLoader.DEFAULT_CFG_PATH)["PP-OCRv6_rec_small"])
    model.load_state_dict(state_dict or torch.load(WEIGHTS_DIR / "PP-OCRv6_rec_small.pth", map_location="cpu",
                                                   weights_only=True))
    with open(WEIGHTS_DIR / "ppocrv6_dict.txt", "rb") as f:
        chars = [line.decode("utf-8").strip("\n").strip("\r\n") for line in f]
    return model.to(device).eval(), ["blank"] + chars + [" "]


def ctc_logits(model, x):
    """Raw CTC scores (batch x steps x classes) through backbone -> CTC encoder -> CTC head."""
    return model.head.ctc_head(model.head.ctc_encoder(model.backbone(x)))


def set_finetune_mode(model):
    """Train mode, but with inference-style pooling (variable width) and frozen batch-norm statistics."""
    model.train()
    model.backbone.training = False            # the training branch pools every line to a fixed 40 steps
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def preprocess(images: list) -> torch.Tensor:
    """RapidOCR's resize_norm_img for a batch of BGR crops."""
    ratio = max([MIN_RATIO] + [im.shape[1] / im.shape[0] for im in images])
    width = int(HEIGHT * ratio)
    batch = np.zeros((len(images), 3, HEIGHT, width), np.float32)
    for i, im in enumerate(images):
        w = min(width, int(math.ceil(HEIGHT * im.shape[1] / im.shape[0])))
        resized = cv2.resize(im, (w, HEIGHT)).astype(np.float32).transpose(2, 0, 1) / 255
        batch[i, :, :, :w] = (resized - 0.5) / 0.5
    return torch.from_numpy(batch)


def decode(probs: torch.Tensor, chars: list) -> list:
    out = []
    for seq in probs.argmax(-1).tolist():
        text, prev = [], 0
        for c in seq:
            if c != prev and c != 0:
                text.append(chars[c])
            prev = c
        out.append("".join(text))
    return out


@torch.no_grad()
def read(model, chars, images: list, device, batch_size: int = 16) -> list:
    """Readings, cleaned exactly like the stored RapidOCR readings (clean_text + heading removed)."""
    model.eval()
    order = np.argsort([im.shape[1] / im.shape[0] for im in images])     # like RapidOCR: similar widths together
    texts = [None] * len(images)
    for s in range(0, len(images), batch_size):
        idx = order[s:s + batch_size]
        probs = model(preprocess([images[i] for i in idx]).to(device)).cpu()
        for i, t in zip(idx, decode(probs, chars)):
            texts[i] = strip_header(clean_text(t))
    return texts


# ---------------------------------------------------------------------------------------- colour crops
def color_crop(image: np.ndarray, quad) -> np.ndarray:
    quad = np.asarray(quad, np.float32)
    w = int(round(max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3]))))
    h = int(round(max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1]))))
    if w < 8 or h < 6:
        return None
    target = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(quad, target), (w, h),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def recut_color_crops(rows: pd.DataFrame):
    """Save colour full-resolution versions of the labelled crops, using the same box order as real_photos.py."""
    COLOR_DIR.mkdir(parents=True, exist_ok=True)
    wanted = set(rows["crop"])
    photos = pd.read_csv(project_path("data/images/packets.csv"), dtype=str)
    for photo in photos[photos["product_id"].isin(rows["product_id"])].itertuples():
        detections, image = detect_and_read(str(project_path(photo.image)))
        k = 0
        for quad, _, _ in detections:
            q = np.asarray(quad, float)
            if np.linalg.norm(q[3] - q[0]) > 1.5 * np.linalg.norm(q[1] - q[0]):
                continue
            crop = color_crop(image, quad)
            if crop is None:
                continue
            name = f"{photo.product_id}_{k:03d}.png"
            if name in wanted:
                cv2.imwrite(str(COLOR_DIR / name), crop)
            k += 1
    missing = wanted - {p.name for p in COLOR_DIR.glob("*.png")}
    if missing:
        raise RuntimeError(f"{len(missing)} crops could not be re-cut (detection changed?)")


# ---------------------------------------------------------------------------------------- fine-tuning
def augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    img = img.astype(np.float32) * rng.uniform(0.75, 1.25) + rng.uniform(-25, 25)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.2))
    h, w = img.shape[:2]
    img = cv2.resize(img, (max(8, round(w * rng.uniform(0.9, 1.1))), h))
    pad = [rng.randint(0, max(1, h // 8)) for _ in range(4)]                   # small crop-box jitter
    img = cv2.copyMakeBorder(img, pad[0], pad[1], pad[2], pad[3], cv2.BORDER_REPLICATE)
    img += np.random.default_rng(rng.randrange(2**31)).normal(0, rng.uniform(0, 5), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


LRS = (1e-5, 3e-5, 1e-4)        # learning rates tried on the inner validation split
MAX_STEPS = 400                 # schedule length; a run may stop earlier at the step chosen on validation
EVAL_EVERY = 50
END_MISMATCH = 2                # characters of unmatched text at a line's start or end = misaligned label


def misaligned(label: str, reading: str) -> bool:
    """True if the crop shows clearly more or less text than its label at the START or END of the line (extra
    heading fragment, neighbouring word, cut-off end). Misreadings inside the line do not count: those are what
    fine-tuning should learn from. Judged with RapidOCR's reading, so no test information is used."""
    if not reading:
        return True
    ops = Levenshtein.opcodes(label.lower(), reading.lower())
    ends = [ops[0], ops[-1]]
    return any(op.tag in ("insert", "delete") and max(op.src_end - op.src_start, op.dest_end - op.dest_start)
               >= END_MISMATCH for op in ends)


def finetune(train: list, chars: list, device, steps: int, lr: float, batch_size: int, seed: int,
             validation: list = None, eval_every: int = EVAL_EVERY):
    """Fine-tune for `steps` steps of a MAX_STEPS one-cycle schedule (so a run stopped early at step s is the
    same as the first s steps of a full run). With `validation`, returns (model, [(step, val CER), ...])."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model, _ = load_ppocr(device)
    index = {c: i for i, c in enumerate(chars) if i > 0}
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=lr, total_steps=max(steps, MAX_STEPS),
                                                   pct_start=0.1)
    curve = []
    for step in range(steps):
        set_finetune_mode(model)
        batch = [rng.choice(train) for _ in range(batch_size)]
        x = preprocess([augment(img, rng) for img, _ in batch]).to(device)
        targets = [[index[c] for c in label if c in index] for _, label in batch]
        logits = ctc_logits(model, x)
        log_probs = logits.log_softmax(-1).permute(1, 0, 2)
        loss = ctc(log_probs, torch.tensor([i for t in targets for i in t], device=device),
                   torch.full((len(batch),), log_probs.shape[0], device=device),
                   torch.tensor([len(t) for t in targets], device=device))
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
        schedule.step()
        if validation and (step + 1) % eval_every == 0:
            preds = read(model, chars, [img for img, _ in validation], device)
            curve.append((step + 1, error_rates(preds, [l for _, l in validation])["cer"]))
    model.eval()
    return (model, curve) if validation else model


def choose_setting(train: list, validation: list, chars: list, device, batch_size: int, seed: int) -> dict:
    """Pick (learning rate, steps) with the lowest CER on the inner validation photos. Step 0 (no fine-tuning,
    the pretrained model) is always a candidate, so fine-tuning is only used if it actually helps."""
    model, _ = load_ppocr(device)
    base = error_rates(read(model, chars, [img for img, _ in validation], device), [l for _, l in validation])["cer"]
    del model
    candidates = [(base, 0, 0.0)]
    curves = {}
    for lr in LRS:
        _, curve = finetune(train, chars, device, MAX_STEPS, lr, batch_size, seed, validation)
        curves[lr] = curve
        candidates += [(cer, step, lr) for step, cer in curve]
        torch.cuda.empty_cache()
        print(f"    lr {lr:g}: val CER " + " ".join(f"{s}:{c:.3f}" for s, c in curve) + f"  (pretrained {base:.3f})",
              flush=True)
    cer, steps, lr = min(candidates, key=lambda c: (round(c[0], 6), c[1]))
    return {"lr": lr, "steps": steps, "val_cer": cer, "val_cer_pretrained": base,
            "curves": {str(k): v for k, v in curves.items()}}


# ---------------------------------------------------------------------------------------- run
def summarise(readings: dict, labels: list) -> dict:
    out = {}
    for name, preds in readings.items():
        exact = error_rates(preds, labels)
        nocase = error_rates([p.lower() for p in preds], [l.lower() for l in labels])
        out[name] = {"cer": exact["cer"], "wer": exact["wer"], "line_accuracy": exact["line_accuracy"],
                     "cer_ignoring_case": nocase["cer"], "wer_ignoring_case": nocase["wer"]}
    return out


def run(batch_size: int = 8, seed: int = 0) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_rows()
    if not all((COLOR_DIR / c).exists() for c in rows["crop"]):
        print("re-cutting colour crops ...", flush=True)
        recut_color_crops(rows)
    images = {c: cv2.imread(str(COLOR_DIR / c)) for c in rows["crop"]}
    labels = rows["label"].tolist()

    model, chars = load_ppocr(device)
    pretrained = read(model, chars, [images[c] for c in rows["crop"]], device)
    same = sum(a == b for a, b in zip(pretrained, rows["rapidocr"]))
    print(f"check: PyTorch copy agrees exactly with RapidOCR on {same}/{len(rows)} crops", flush=True)

    halves = photo_halves(rows, seed)
    finetuned = dict()
    folds = []
    for k, (train_photos, test_photos) in enumerate([(halves[0], halves[1]), (halves[1], halves[0])]):
        tr = rows[rows["product_id"].isin(train_photos)]
        te = rows[rows["product_id"].isin(test_photos)]           # test half: never filtered, never looked at
        keep = [not misaligned(l, r) for l, r in zip(tr["label"], tr["rapidocr"])]
        clean = tr[keep]
        inner = sorted(clean["product_id"].unique())
        random.Random(seed + k).shuffle(inner)
        val_photos = set(inner[:max(1, len(inner) // 4)])
        items = lambda d: [(images[c], l) for c, l in zip(d["crop"], d["label"])]
        print(f"fold {k}: {len(tr)} training crops, {len(tr) - len(clean)} dropped as misaligned; inner validation "
              f"on {len(val_photos)} photos; test on {len(te)} crops", flush=True)
        setting = choose_setting(items(clean[~clean["product_id"].isin(val_photos)]),
                                 items(clean[clean["product_id"].isin(val_photos)]), chars, device, batch_size, seed + k)
        print(f"  chosen: lr {setting['lr']:g}, {setting['steps']} steps (val CER {setting['val_cer']:.3f}, "
              f"pretrained {setting['val_cer_pretrained']:.3f})", flush=True)
        if setting["steps"]:
            model = finetune(items(clean), chars, device, setting["steps"], setting["lr"], batch_size, seed + k)
        else:
            model, _ = load_ppocr(device)                          # validation said: do not fine-tune
        out = project_path(f"models/runs/ocr/ppocr_finetuned_fold{k}")
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "fold": k, "train_photos": sorted(train_photos)}, out / "model.pt")
        preds = read(model, chars, [images[c] for c in te["crop"]], device)
        finetuned.update(zip(te["crop"], preds))
        torch.cuda.empty_cache()
        folds.append({"fold": k, "train_crops": len(tr), "dropped_misaligned": len(tr) - len(clean),
                      "chosen": setting, "test_crops": len(te),
                      "test_cer_before": error_rates([pretrained[i] for i in np.flatnonzero(rows["product_id"].isin(test_photos))],
                                                     te["label"].tolist())["cer"],
                      "test_cer_after": error_rates(preds, te["label"].tolist())["cer"]})

    ours = json.loads(project_path("reports/ocr_model/crnn_vs_rapidocr.json").read_text()) \
        if project_path("reports/ocr_model/crnn_vs_rapidocr.json").exists() else None
    readings = {"rapidocr": rows["rapidocr"].tolist(), "ppocr_torch_pretrained": pretrained,
                "ppocr_finetuned": [finetuned[c] for c in rows["crop"]]}
    result = {"crops": len(rows), "photos": rows["product_id"].nunique(),
              "check_exact_agreement_with_rapidocr": same,
              "settings": {"learning_rates_tried": LRS, "max_steps": MAX_STEPS, "eval_every": EVAL_EVERY,
                           "batch_size": batch_size, "seed": seed, "end_mismatch_chars": END_MISMATCH},
              "folds": folds,
              "systems": summarise(readings, labels),
              "our_crnn_finetuned": ours["summary"]["crnn_ft"] if ours else None}
    per_photo = {pid: {n: error_rates([readings[n][i] for i in g.index], g["label"].tolist())["cer"]
                       for n in ("rapidocr", "ppocr_finetuned")} for pid, g in rows.groupby("product_id")}
    result["photos_finetuned_better"] = sum(v["ppocr_finetuned"] < v["rapidocr"] for v in per_photo.values())
    result["photos_rapidocr_better"] = sum(v["rapidocr"] < v["ppocr_finetuned"] for v in per_photo.values())
    result["examples"] = [{"label": l, "before": a, "after": b} for l, a, b in
                          zip(labels, readings["rapidocr"], readings["ppocr_finetuned"]) if a != b][:30]
    project_path("reports/ocr_model/ppocr_finetune.json").write_text(json.dumps(result, indent=2))
    report(result)
    return result


def report(result: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = result["systems"]
    lines = ["# Fine-tuning RapidOCR's recogniser (PP-OCRv6 small) on food-label crops", "",
             f"{result['crops']} real crops from {result['photos']} photos, two-fold split by photo. PyTorch copy "
             f"agrees exactly with RapidOCR on {result['check_exact_agreement_with_rapidocr']}/{result['crops']} crops "
             "before fine-tuning.", "",
             "| Recogniser | CER | CER (ignoring case) | WER | Lines exact |", "|---|---|---|---|---|"]
    names = {"rapidocr": "RapidOCR as shipped", "ppocr_torch_pretrained": "Same model in PyTorch (our colour crops)",
             "ppocr_finetuned": "RapidOCR recogniser fine-tuned on food labels"}
    for k, m in s.items():
        lines.append(f"| {names[k]} | {m['cer']:.3f} | {m['cer_ignoring_case']:.3f} | {m['wer']:.3f} | {m['line_accuracy']:.1%} |")
    if result["our_crnn_finetuned"]:
        m = result["our_crnn_finetuned"]
        lines.append(f"| Our CRNN v2 + fine-tuning (from scratch) | {m['cer']:.3f} | {m['cer_ignoring_case']:.3f} | "
                     f"{m['wer']:.3f} | {m['line_accuracy']:.1%} |")
    lines += ["", f"Per photo: fine-tuned better on {result['photos_finetuned_better']}, RapidOCR better on "
              f"{result['photos_rapidocr_better']}.", "",
              "Folds (learning rate and steps chosen on validation photos inside the training half; "
              "misaligned training crops dropped; test halves unchanged):", ""]
    lines += [f"- fold {f['fold']}: {f['dropped_misaligned']} of {f['train_crops']} training crops dropped; chosen "
              f"lr {f['chosen']['lr']:g}, {f['chosen']['steps']} steps (validation CER {f['chosen']['val_cer']:.3f} "
              f"vs pretrained {f['chosen']['val_cer_pretrained']:.3f}); test CER {f['test_cer_before']:.3f} -> "
              f"{f['test_cer_after']:.3f}" for f in result["folds"]]
    lines += ["",
              "## Readings changed by fine-tuning", "", "| Label | RapidOCR | Fine-tuned |", "|---|---|---|"]
    lines += [f"| `{e['label']}` | `{e['before']}` | `{e['after']}` |" for e in result["examples"]]
    project_path("reports/ocr_model/ppocr_finetune.md").write_text("\n".join(lines), encoding="utf-8")

    bars = [("Our CRNN\n(from scratch + fine-tuned)", result["our_crnn_finetuned"], "#C9B8A6"),
            ("RapidOCR\n(pretrained, as shipped)", s["rapidocr"], "#9FB9B2"),
            ("RapidOCR recogniser\nfine-tuned on food labels", s["ppocr_finetuned"], "#0E7C66")]
    bars = [b for b in bars if b[1]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, key, title in [(axes[0], "cer", "Character error rate"), (axes[1], "cer_ignoring_case", "CER ignoring case")]:
        h = ax.bar([b[0] for b in bars], [b[1][key] for b in bars], color=[b[2] for b in bars])
        ax.bar_label(h, fmt="%.3f", fontsize=8, padding=2)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Real food-label crops ({result['crops']} crops, {result['photos']} photos, two-fold by photo)", fontsize=11)
    fig.tight_layout()
    fig.savefig(project_path("reports/figures/ocr_model/ocr_all_models.png"), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    a = parser.parse_args()
    r = run(a.batch_size, a.seed)
    for k, m in r["systems"].items():
        print(f"{k:24s} CER {m['cer']:.3f}  CER(no case) {m['cer_ignoring_case']:.3f}  exact {m['line_accuracy']:.1%}")


if __name__ == "__main__":
    main()
