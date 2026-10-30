"""Two ways to improve our CRNN on real packet crops, measured on the same 377 test crops as before.

    python -m src.ocr_model.improve_crnn

1. More real training data: fine-tune CRNN v2 (crnn_tough) on the training-half crops PLUS the Open Food Facts
   crops from collect_off_crops.py (train/validation-split products only, misaligned ones dropped), mixed with
   synthetic lines as before. Same two-fold split by photo as finetune.py.
2. Smarter decoding: CTC prefix beam search with our food character language model (combine.CharLM, trained on
   train-split ingredient lists only) instead of greedy decoding. The language-model weight and length bonus
   are cross-fitted: chosen on one photo half using the fold model that never saw that half, then applied to
   the other half, so no test crop influences its own setting.
Writes reports/ocr_model/improve_crnn.json/.md and reports/figures/ocr_model/crnn_improvements.png.
"""
import json
import math

import cv2
import numpy as np
import pandas as pd
import torch

from src.ocr_model.charset import BLANK, ID2CHAR
from src.ocr_model.evaluate import error_rates
from src.ocr_model.finetune import crnn_read, finetune, load_rows, photo_halves, read_crop
from src.ocr_model.ppocr_finetune import misaligned
from src.ocr_model.real_photos import load_model, pad_width, strip_header
from src.ocr_model.render_lines import HEIGHT
from src.utils.config import project_path

OFF_DIR = project_path("data/ocr_crops/off")
ALPHAS = (0.0, 0.2, 0.4, 0.7, 1.0)      # language-model weight (0 = beam search without the LM)
BETAS = (0.0, 0.5, 1.0)                 # bonus per output character (offsets the LM's preference for short text)
BEAM, TOP_K = 8, 6


# ---------------------------------------------------------------------------------------- data
def off_items() -> list:
    """Open Food Facts crops as grey 32-px images with labels; misaligned ones dropped."""
    path = OFF_DIR / "labels.csv"
    if not path.exists():
        return []
    rows = pd.read_csv(path, dtype=str, keep_default_na=False).drop_duplicates("crop")
    rows = rows[[not misaligned(l, r) for l, r in zip(rows["label"], rows["rapidocr"])]]
    items = []
    for r in rows.itertuples():
        img = cv2.imread(str(OFF_DIR / r.crop), cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape[0] < 6:
            continue
        w = max(4, round(img.shape[1] * HEIGHT / img.shape[0]))
        items.append((cv2.resize(img, (w, HEIGHT), interpolation=cv2.INTER_AREA if img.shape[0] > HEIGHT
                                 else cv2.INTER_CUBIC), r.label))
    return items


# ---------------------------------------------------------------------------------------- beam search
@torch.no_grad()
def log_probs_of(model, images: list, device) -> list:
    out = []
    for img in images:
        x = torch.from_numpy(pad_width(img)).float().div(255).view(1, 1, HEIGHT, -1).to(device)
        out.append(model(x)[:, 0].float().cpu().numpy())          # T x classes, log-probabilities
    return out


def _logadd(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def beam_decode(lp: np.ndarray, lm, alpha: float, beta: float, beam: int = BEAM, top_k: int = TOP_K) -> str:
    """CTC prefix beam search with shallow fusion: score = log P_ctc + alpha * log P_lm + beta * length."""
    beams = {"": (0.0, -math.inf)}                      # prefix -> (log P ending in blank, log P ending in non-blank)
    lm_cache = {"": 0.0}

    def lm_logp(prefix):
        if prefix not in lm_cache:
            parent = prefix[:-1]
            history = "\x02" * 5 + parent.lower()
            lm_cache[prefix] = lm_logp(parent) + math.log(max(lm.prob(history, prefix[-1].lower()), 1e-12))
        return lm_cache[prefix]

    for t in range(lp.shape[0]):
        row = lp[t]
        candidates = np.argsort(row)[-top_k:]
        nxt = {}
        for prefix, (pb, pnb) in beams.items():
            total = _logadd(pb, pnb)
            for c in candidates:
                p = float(row[c])
                if c == BLANK:
                    b, nb = nxt.get(prefix, (-math.inf, -math.inf))
                    nxt[prefix] = (_logadd(b, total + p), nb)
                    continue
                ch = ID2CHAR.get(int(c), "")
                new = prefix + ch
                b, nb = nxt.get(new, (-math.inf, -math.inf))
                if prefix and prefix[-1] == ch:             # repeated character needs a blank in between
                    nxt[new] = (b, _logadd(nb, pb + p))
                    sb, snb = nxt.get(prefix, (-math.inf, -math.inf))
                    nxt[prefix] = (sb, _logadd(snb, pnb + p))
                else:
                    nxt[new] = (b, _logadd(nb, total + p))

        def score(item):
            prefix, (pb, pnb) = item
            return _logadd(pb, pnb) + alpha * lm_logp(prefix) + beta * len(prefix)
        beams = dict(sorted(nxt.items(), key=score, reverse=True)[:beam])
    best = max(beams.items(), key=lambda it: _logadd(*it[1]) + alpha * lm_logp(it[0]) + beta * len(it[0]))
    return best[0]


# ---------------------------------------------------------------------------------------- run
def run(base: str = "crnn_tough", steps: int = 1500, lr: float = 3e-4, seed: int = 0) -> dict:
    from src.ocr_model.combine import resources
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_rows()
    labels = rows["label"].tolist()
    halves = photo_halves(rows, seed)
    extra = off_items()
    print(f"Open Food Facts crops used for training: {len(extra)}", flush=True)
    test_images = {c: read_crop(c) for c in rows["crop"]}

    greedy_before = json.loads(project_path("reports/ocr_model/crnn_vs_rapidocr.json").read_text())["summary"]
    fold_models, greedy_after, fold_info = {}, {}, []
    for k, (train_photos, test_photos) in enumerate([(halves[0], halves[1]), (halves[1], halves[0])]):
        tr = rows[rows["product_id"].isin(train_photos)]
        te = rows[rows["product_id"].isin(test_photos)]
        print(f"fold {k}: fine-tune on {len(tr)} test-set-half crops + {len(extra)} Open Food Facts crops", flush=True)
        model = finetune(base, tr, steps, lr, 16, 16, seed + k, device, extra=extra)
        out = project_path(f"models/runs/ocr/{base}_off_fold{k}")
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"model": "crnn", "state_dict": model.state_dict(), "base": base, "fold": k,
                    "off_crops": len(extra)}, out / "best.pt")
        fold_models[k] = (model, te)
        preds = crnn_read(model, [test_images[c] for c in te["crop"]], device)
        greedy_after.update(zip(te["crop"], preds))
        fold_info.append({"fold": k, "test_cer_greedy": error_rates(preds, te["label"].tolist())["cer"]})

    # beam search + LM: log-probabilities of every crop from the fold model that never saw its photo
    lm, _ = resources()
    lps = {}
    for k, (model, te) in fold_models.items():
        lps.update(zip(te["crop"], log_probs_of(model, [test_images[c] for c in te["crop"]], device)))
    beam_preds, chosen = {}, []
    for choose_half, apply_half in [(halves[0], halves[1]), (halves[1], halves[0])]:
        pick = rows[rows["product_id"].isin(choose_half)]
        grid = {}
        for a in ALPHAS:
            for b in BETAS:
                preds = [strip_header(beam_decode(lps[c], lm, a, b)) for c in pick["crop"]]
                grid[(a, b)] = error_rates(preds, pick["label"].tolist())["cer"]
        a, b = min(grid, key=grid.get)
        chosen.append({"alpha": a, "beta": b, "cer_on_choice_half": grid[(a, b)],
                       "greedy_like_cer_on_choice_half": grid[(0.0, 0.0)]})
        use = rows[rows["product_id"].isin(apply_half)]
        beam_preds.update(zip(use["crop"], [strip_header(beam_decode(lps[c], lm, a, b)) for c in use["crop"]]))
        print(f"  beam setting chosen on one half: alpha {a}, beta {b}", flush=True)

    systems = {}
    for name, preds in [("CRNN v2 + fine-tuned (before)", None),
                        ("+ Open Food Facts crops", [greedy_after[c] for c in rows["crop"]]),
                        ("+ Open Food Facts crops + beam search with food LM", [beam_preds[c] for c in rows["crop"]])]:
        if preds is None:
            systems[name] = {k: greedy_before["crnn_ft"][k] for k in ("cer", "cer_ignoring_case", "wer", "line_accuracy")}
            continue
        exact = error_rates(preds, labels)
        nocase = error_rates([p.lower() for p in preds], [l.lower() for l in labels])
        systems[name] = {"cer": exact["cer"], "cer_ignoring_case": nocase["cer"], "wer": exact["wer"],
                         "line_accuracy": exact["line_accuracy"]}
    systems["RapidOCR (reference)"] = {k: greedy_before["rapidocr"][k] for k in ("cer", "cer_ignoring_case", "wer", "line_accuracy")}
    result = {"crops": len(rows), "off_training_crops": len(extra), "folds": fold_info, "beam_settings": chosen,
              "systems": systems,
              "examples": [{"label": l, "greedy": greedy_after[c], "beam": beam_preds[c]}
                           for c, l in zip(rows["crop"], labels) if greedy_after[c] != beam_preds[c]][:20]}
    suffix = "" if base == "crnn_tough" else f"_{base}"
    result["base"] = base
    project_path(f"reports/ocr_model/improve_crnn{suffix}.json").write_text(json.dumps(result, indent=2, default=str))
    report(result, suffix)
    return result


def report(result: dict, suffix: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lines = ["# Improving our CRNN on real packet crops", "",
             f"Same {result['crops']} real test crops, two-fold split by photo. "
             f"{result['off_training_crops']} Open Food Facts crops added to training.", "",
             "| Reader | CER | CER (ignoring case) | WER | Lines exact |", "|---|---|---|---|---|"]
    for name, m in result["systems"].items():
        lines.append(f"| {name} | {m['cer']:.3f} | {m['cer_ignoring_case']:.3f} | {m['wer']:.3f} | {m['line_accuracy']:.1%} |")
    lines += ["", "Beam-search settings (chosen on the other half): " +
              "; ".join(f"alpha {c['alpha']}, beta {c['beta']}" for c in result["beam_settings"]), "",
              "## Lines changed by beam search", "", "| Label | Greedy | Beam + LM |", "|---|---|---|"]
    lines += [f"| `{e['label']}` | `{e['greedy']}` | `{e['beam']}` |" for e in result["examples"]]
    project_path(f"reports/ocr_model/improve_crnn{suffix}.md").write_text("\n".join(lines), encoding="utf-8")

    names = list(result["systems"])
    vals = [result["systems"][n]["cer"] for n in names]
    colours = ["#C9B8A6", "#5FA898", "#0E7C66", "#9FB9B2"]
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    bars = ax.barh(names[::-1], vals[::-1], color=colours[:len(names)][::-1])
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xlabel("character error rate on 377 real crops (lower is better)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(project_path(f"reports/figures/ocr_model/crnn_improvements{suffix}.png"), dpi=160)
    plt.close(fig)


def improved_readings(base: str = "crnn_tough", seed: int = 0) -> pd.DataFrame:
    """Per-crop readings of the improved CRNN (Open Food Facts fine-tuning + beam search), rebuilt from the saved
    fold models and the beam settings in improve_crnn.json: every crop is read by the fold model that never saw
    its photo, with the beam setting chosen on the other half. Cached in reports/ocr_model/improved_crnn_readings.csv."""
    from src.ocr_model.combine import resources
    suffix = "" if base == "crnn_tough" else f"_{base}"
    cache = project_path(f"reports/ocr_model/improved_crnn_readings{suffix}.csv")
    rows = load_rows()
    if cache.exists():
        return rows.merge(pd.read_csv(cache, dtype=str, keep_default_na=False), on="crop")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings = json.loads(project_path(f"reports/ocr_model/improve_crnn{suffix}.json").read_text())["beam_settings"]
    halves = photo_halves(rows, seed)
    lm, _ = resources()
    readings = {}
    # fold k was trained on halves[k] and tests halves[1-k]; beam setting i was chosen on halves[i], applied to halves[1-i]
    for k, test_half in enumerate([halves[1], halves[0]]):
        model = load_model(f"{base}_off_fold{k}", device)
        te = rows[rows["product_id"].isin(test_half)]
        s = settings[k]
        lps = log_probs_of(model, [read_crop(c) for c in te["crop"]], device)
        readings.update(zip(te["crop"], [strip_header(beam_decode(lp, lm, s["alpha"], s["beta"])) for lp in lps]))
    out = pd.DataFrame({"crop": list(readings), "crnn_improved": list(readings.values())})
    out.to_csv(cache, index=False)
    return rows.merge(out, on="crop")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="crnn_tough", help="synthetic-trained CRNN run to start from")
    r = run(parser.parse_args().base)
    for name, m in r["systems"].items():
        print(f"{name:52s} CER {m['cer']:.4f}  no-case {m['cer_ignoring_case']:.4f}  exact {m['line_accuracy']:.1%}")


if __name__ == "__main__":
    main()
