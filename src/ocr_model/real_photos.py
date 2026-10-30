"""Real-photo test: our CRNN vs RapidOCR's recogniser on the SAME text-line crops of our 60 packet photos.

    python -m src.ocr_model.real_photos                    # crops + labels + comparison
    python -m src.ocr_model.real_photos --run crnn

How the test set is made (data/images/packets.csv has the typed ingredient list of every photo):
  1. Each photo is prepared exactly as in the app (straightened, rotated upright, small print enlarged).
  2. RapidOCR's text DETECTOR finds the text boxes. Each box is cut out and rectified to a straight line image.
     Both recognisers read these identical crops, so the comparison is recogniser against recogniser.
  3. Label: the part of the typed ingredient list the crop shows. Both readings are aligned to the typed list
     (best-matching substring by edit distance); the label is the UNION of the two aligned spans, widened to
     whole words. Using both readings means neither model's mistakes shape the label (a model that drops the
     last word does not get a label without that word, because the other reading still covers it).
  4. Crops that match the ingredient list badly in both readings (other parts of the pack: brand name,
     nutrition table, address) are left out, as are vertical text boxes and readings under 8 letters/digits
     (too short to place reliably). A leading "Ingredients:" heading is removed from both readings, because
     the typed lists do not contain it.
Outputs: data/ocr_crops/real/*.png + labels.csv, reports/ocr_model/real_photos.json/.md, a sample sheet.
"""
import argparse
import csv
import json
import re

import cv2
import numpy as np
import pandas as pd
import torch
from rapidfuzz import fuzz

from src.ocr_model.charset import clean_text
from src.ocr_model.evaluate import error_rates
from src.ocr_model.models import build, greedy_decode
from src.ocr_model.render_lines import HEIGHT, STRIDE
from src.utils.config import project_path

CROP_DIR = project_path("data/ocr_crops/real")
MIN_MATCH = 0.75         # aligned edit similarity a reading needs to count as "this crop shows the list"
MIN_CHARS = 8            # shorter readings ("Per", "16%") match some random place in a long list
HEADER_RE = re.compile(r"^\s*ingredients?\s*[:;.\-]?\s*", re.IGNORECASE)


def strip_header(text: str) -> str:
    """'INGREDIENTS: Sugar, ...' -> 'Sugar, ...'. The heading is printed on the pack but is not part of the
    typed ingredient list, so it is removed from BOTH readings before labelling and scoring."""
    return HEADER_RE.sub("", text)


# ---------------------------------------------------------------------------------------- crops
def detect_and_read(path: str) -> list:
    """[(quad box, RapidOCR text, confidence), ...] on the photo as the app prepares it, plus that image."""
    from src.ocr.ocr_pipeline import get_rapidocr, prepare_and_read, preprocess
    image, _, _, scale = prepare_and_read(path)
    image = preprocess(image, scale)
    out = get_rapidocr()(image, use_det=True, use_cls=False, use_rec=True, text_score=0.0)
    if out.boxes is None:
        return [], image
    return list(zip(out.boxes.tolist(), out.txts, out.scores)), image


def rectify(image: np.ndarray, quad) -> np.ndarray:
    """Cut a (possibly tilted) text box out of the photo as a straight grey line image of height 32."""
    quad = np.asarray(quad, np.float32)
    w = int(round(max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3]))))
    h = int(round(max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1]))))
    if w < 8 or h < 6:
        return None
    target = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    crop = cv2.warpPerspective(image, cv2.getPerspectiveTransform(quad, target), (w, h), flags=cv2.INTER_CUBIC)
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    new_w = max(STRIDE, round(w * HEIGHT / h))
    return cv2.resize(grey, (new_w, HEIGHT), interpolation=cv2.INTER_AREA if h > HEIGHT else cv2.INTER_CUBIC)


def pad_width(img: np.ndarray, text_len: int = 0) -> np.ndarray:
    """Width a multiple of STRIDE and long enough for CTC (no upper limit: the models accept any width)."""
    need = max(img.shape[1], STRIDE * (text_len + 2))
    need = -(-need // STRIDE) * STRIDE
    return img if need == img.shape[1] else cv2.resize(img, (need, HEIGHT), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------------------- labels
def best_span(reading: str, reference: str) -> tuple:
    """(start, end, similarity) of the reference substring closest to `reading` by edit distance
    (semi-global alignment: free start and end in the reference). Similarity = 1 - distance / len(reading)."""
    if not reading:
        return 0, 0, 0.0
    coarse = fuzz.partial_ratio_alignment(reading.lower(), reference.lower())
    lo = max(0, coarse.dest_start - 15)
    hi = min(len(reference), coarse.dest_end + 15)
    a, b = reading.lower(), reference[lo:hi].lower()
    n, m = len(a), len(b)
    prev = list(range(n + 1))                    # column-wise DP over the reference: rows = reading positions
    starts_prev = [0] * (n + 1)
    best = (n, 0, 0)                             # (distance, start, end) in b
    for j in range(1, m + 1):
        cur, starts = [0] * (n + 1), [j] * (n + 1)
        cur[0], starts[0] = 0, j                 # free start: alignment may begin at any reference position
        for i in range(1, n + 1):
            options = [(prev[i - 1] + (a[i - 1] != b[j - 1]), starts_prev[i - 1]),
                       (prev[i] + 1, starts_prev[i]),
                       (cur[i - 1] + 1, starts[i - 1])]
            cur[i], starts[i] = min(options)
        if cur[n] < best[0]:
            best = (cur[n], starts[n], j)
        prev, starts_prev = cur, starts
    distance, s, e = best
    return lo + s, lo + e, 1 - distance / n


def widen_to_words(reference: str, start: int, end: int) -> tuple:
    while start > 0 and reference[start - 1] != " ":
        start -= 1
    while end < len(reference) and reference[end] != " ":
        end += 1
    return start, end


def label_crop(readings: list, reference: str):
    """Typed text shown by a crop, from the union of the readings' aligned spans; None if no reading matches."""
    spans = [best_span(r, reference) for r in readings if sum(ch.isalnum() for ch in r) >= MIN_CHARS]
    spans = [s for s in spans if s[2] >= MIN_MATCH]
    if not spans:
        return None
    start, end = widen_to_words(reference, min(s[0] for s in spans), max(s[1] for s in spans))
    label = reference[start:end].strip()
    longest = max(len(r) for r in readings)
    if not label or not 0.5 * longest <= len(label) <= 2 * longest:     # spans far apart: not one place
        return None
    return label


# ---------------------------------------------------------------------------------------- run
def load_model(run: str, device):
    ckpt = torch.load(project_path(f"models/runs/ocr/{run}/best.pt"), map_location=device)
    model = build(ckpt["model"]).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    return model


@torch.no_grad()
def crnn_read(model, images: list, device) -> list:
    out = []
    for img in images:
        x = torch.from_numpy(pad_width(img)).float().div(255).view(1, 1, HEIGHT, -1).to(device)
        out += greedy_decode(model(x).cpu())
    return out


def build_and_compare(run: str = "crnn") -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(run, device)
    photos = pd.read_csv(project_path("data/images/packets.csv"), dtype=str)
    CROP_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for photo in photos.itertuples():
        reference = clean_text(photo.reference_text)
        detections, image = detect_and_read(str(project_path(photo.image)))
        crops, rapid = [], []
        for quad, text, conf in detections:
            q = np.asarray(quad, float)
            if np.linalg.norm(q[3] - q[0]) > 1.5 * np.linalg.norm(q[1] - q[0]):     # vertical text box
                continue
            crop = rectify(image, quad)
            if crop is not None:
                crops.append(crop)
                rapid.append(strip_header(clean_text(text)))
        ours = [strip_header(t) for t in crnn_read(model, crops, device)]
        for k, (crop, r, c) in enumerate(zip(crops, rapid, ours)):
            label = label_crop([r, c], reference)
            if label is None:
                continue
            name = f"{photo.product_id}_{k:03d}.png"
            cv2.imwrite(str(CROP_DIR / name), crop)
            rows.append({"crop": name, "product_id": photo.product_id, "country_group": photo.country_group,
                         "label": label, "rapidocr": r, "crnn": c})
        print(f"{photo.product_id}: {sum(r['product_id'] == photo.product_id for r in rows)} crops", flush=True)

    with open(CROP_DIR / "labels.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summarise(rows, run)


def summarise(rows: list, run: str) -> dict:
    labels = [r["label"] for r in rows]
    result = {"crops": len(rows), "photos": len({r["product_id"] for r in rows}), "run": run, "systems": {}}
    for system in ("rapidocr", "crnn"):
        preds = [r[system] for r in rows]
        exact = error_rates(preds, labels)
        nocase = error_rates([p.lower() for p in preds], [l.lower() for l in labels])
        result["systems"][system] = {"cer": exact["cer"], "wer": exact["wer"], "line_accuracy": exact["line_accuracy"],
                                     "cer_ignoring_case": nocase["cer"], "wer_ignoring_case": nocase["wer"]}
    by_photo = pd.DataFrame(rows).groupby("product_id")
    per_photo = {pid: {s: error_rates(g[s].tolist(), g["label"].tolist())["cer"] for s in ("rapidocr", "crnn")}
                 for pid, g in by_photo}
    result["photos_where_crnn_better"] = sum(v["crnn"] < v["rapidocr"] for v in per_photo.values())
    result["photos_where_rapidocr_better"] = sum(v["rapidocr"] < v["crnn"] for v in per_photo.values())
    result["per_photo_cer"] = per_photo
    result["examples"] = [r for r in rows if r["crnn"] != r["rapidocr"]][:25]
    project_path("reports/ocr_model/real_photos.json").write_text(json.dumps(result, indent=2))
    write_markdown(result)
    return result


def write_markdown(result: dict):
    names = {"rapidocr": "RapidOCR recogniser (PP-OCRv6, pretrained)", "crnn": "Our CRNN (trained on synthetic lines)"}
    lines = ["# Real packet photos: our CRNN vs RapidOCR", "",
             f"{result['crops']} text-line crops from {result['photos']} photos. Both recognisers read the same crops "
             "(found by RapidOCR's detector). Labels come from the typed ingredient lists (see real_photos.py).", "",
             "| Recogniser | CER | WER | CER (ignoring case) | WER (ignoring case) | Lines exact |", "|---|---|---|---|---|---|"]
    for s, m in result["systems"].items():
        lines.append(f"| {names[s]} | {m['cer']:.3f} | {m['wer']:.3f} | {m['cer_ignoring_case']:.3f} | "
                     f"{m['wer_ignoring_case']:.3f} | {m['line_accuracy']:.1%} |")
    lines += ["", f"Per photo (character error rate): CRNN better on {result['photos_where_crnn_better']} photos, "
              f"RapidOCR better on {result['photos_where_rapidocr_better']}.", "",
              "## Crops where the two disagree", "", "| Label | RapidOCR | CRNN |", "|---|---|---|"]
    lines += [f"| `{e['label']}` | `{e['rapidocr']}` | `{e['crnn']}` |" for e in result["examples"]]
    project_path("reports/ocr_model/real_photos.md").write_text("\n".join(lines), encoding="utf-8")


def sample_sheet(n: int = 24, seed: int = 0):
    """Crops with their label and both readings, for checking the labels by eye."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = pd.read_csv(CROP_DIR / "labels.csv", dtype=str, keep_default_na=False).sample(n, random_state=seed)
    fig, axes = plt.subplots(n, 1, figsize=(10, 0.95 * n))
    for ax, r in zip(axes, rows.itertuples()):
        ax.imshow(cv2.imread(str(CROP_DIR / r.crop), cv2.IMREAD_GRAYSCALE), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"label: {r.label}\nRapidOCR: {r.rapidocr}   |   CRNN: {r.crnn}", fontsize=6.5, loc="left",
                     family="monospace", pad=2)
        ax.axis("off")
    fig.tight_layout()
    path = project_path("reports/figures/ocr_model/real_crops_sample.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="crnn")
    result = build_and_compare(parser.parse_args().run)
    for s, m in result["systems"].items():
        print(f"{s:9s} CER {m['cer']:.3f}  WER {m['wer']:.3f}  CER(no case) {m['cer_ignoring_case']:.3f}  "
              f"exact lines {m['line_accuracy']:.1%}")
    print("sample sheet:", sample_sheet())


if __name__ == "__main__":
    main()
