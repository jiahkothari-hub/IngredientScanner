"""Real training crops at scale: Open Food Facts ingredient photos, cut into labelled text lines.

    python -m src.ocr_model.collect_off_crops --photos 3000          # resumable: rerun to continue

Products come from the silver TRAIN and VALIDATION splits only; the 60 test photos are all in the TEST split,
and any product in data/images/packets.csv is skipped as well, so no test product is ever used for training.
For each product:
  1. ask the Open Food Facts API for the product's selected ENGLISH ingredient photo (ingredients_en);
  2. download the full-size photo, scale it to at most 1600 px (as the test photos were) - the photo itself is
     NOT kept;
  3. prepare and detect it exactly as the app does, cut every horizontal text box out in colour;
  4. label each box with the matching part of the product's typed ingredient list, the same alignment as the
     test set (real_photos.label_crop, here with RapidOCR's reading only); unmatched boxes are discarded.
Markets are taken in turn (India, UK & Ireland, other English, North America) so no market dominates.
Requests are paced (API ~1 per 1.5 s, backing off on HTTP 429) and identify the project.
Photos: Open Food Facts contributors, CC BY-SA 3.0. Text: ODbL.
Outputs: data/ocr_crops/off/<product>_<k>.jpg, data/ocr_crops/off/labels.csv, progress in progress.json.
"""
import argparse
import csv
import gzip
import json
import random
import time
from collections import deque

import cv2
import numpy as np
import pandas as pd
import requests

from src.ocr_model.charset import clean_text
from src.ocr_model.real_photos import label_crop, strip_header
from src.utils.config import project_path

OUT_DIR = project_path("data/ocr_crops/off")
API = "https://world.openfoodfacts.org/api/v2/product/{}"
IMAGES = "https://images.openfoodfacts.org/images/products/{}/{}.jpg"
HEADERS = {"User-Agent": "IngredientScanner-student-project/1.0 (course project; ingredient OCR research)"}
MAX_SIDE = 1600
API_PAUSE = 1.5
FIELDS = ["crop", "product_id", "country_group", "split", "label", "rapidocr", "confidence"]
MARKET_ORDER = ["india", "uk_ireland", "other_english", "north_america"]


def product_path(code: str) -> str:
    """Open Food Facts image folder: 13+ digit codes are split 3/3/3/rest."""
    code = code.zfill(13) if code.isdigit() and len(code) <= 13 else code
    return f"{code[:3]}/{code[3:6]}/{code[6:9]}/{code[9:]}" if len(code) > 8 else code


def candidates(seed: int = 0) -> deque:
    """Train/validation products, shuffled within each market, then interleaved market by market."""
    test_photos = set(pd.read_csv(project_path("data/images/packets.csv"), dtype=str)["product_id"])
    by_market = {m: [] for m in MARKET_ORDER}
    for split in ("train", "validation"):
        with gzip.open(project_path(f"data/splits/silver/{split}.jsonl.gz"), "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["product_id"] not in test_photos:
                    by_market[r["country_group"]].append((r["product_id"], r["country_group"], split, r["text"]))
    rng = random.Random(seed)
    for items in by_market.values():
        rng.shuffle(items)
    order = deque()
    for i in range(max(len(v) for v in by_market.values())):
        for m in MARKET_ORDER:
            if i < len(by_market[m]):
                order.append(by_market[m][i])
    return order


def get(url: str, session, **kw):
    for attempt in range(6):
        try:
            r = session.get(url, headers=HEADERS, timeout=40, **kw)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 429:
            time.sleep(15 * (attempt + 1))
            continue
        return r
    return None


def ingredient_photo(code: str, session):
    r = get(API.format(code), session, params={"fields": "images"})
    if r is None or r.status_code != 200:
        return None
    images = (r.json().get("product") or {}).get("images") or {}
    if "ingredients_en" not in images:
        return None
    imgid = images["ingredients_en"].get("imgid")
    return IMAGES.format(product_path(code), imgid) if imgid else None


def crops_of_photo(photo: np.ndarray, reference: str) -> list:
    """[(colour crop, label, RapidOCR reading, confidence)] for the boxes that match the ingredient list."""
    from src.ocr.ocr_pipeline import get_rapidocr, prepare_and_read, preprocess
    from src.ocr_model.ppocr_finetune import color_crop
    image, _, _, scale = prepare_and_read(photo)
    image = preprocess(image, scale)
    out = get_rapidocr()(image, use_det=True, use_cls=False, use_rec=True, text_score=0.0)
    if out.boxes is None:
        return []
    found = []
    for quad, text, conf in zip(out.boxes.tolist(), out.txts, out.scores):
        q = np.asarray(quad, float)
        if np.linalg.norm(q[3] - q[0]) > 1.5 * np.linalg.norm(q[1] - q[0]):          # vertical text
            continue
        crop = color_crop(image, quad)
        if crop is None:
            continue
        reading = strip_header(clean_text(text))
        label = label_crop([reading], reference)
        if label:
            found.append((crop, label, reading, float(conf)))
    return found


def collect(n_photos: int, seed: int = 0):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    progress_path, labels_path = OUT_DIR / "progress.json", OUT_DIR / "labels.csv"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else \
        {"tried": [], "photos": 0, "no_photo": 0, "failed": 0, "crops": 0, "by_market": {}}
    tried = set(progress["tried"])
    new_file = not labels_path.exists()
    if not new_file:                    # photos written after the last progress save (run stopped mid-way)
        done = pd.read_csv(labels_path, dtype=str)["product_id"].unique()
        progress["photos"] += len(set(done) - tried)
        tried |= set(done)
        progress["tried"] = sorted(tried)
    session = requests.Session()
    queue = candidates(seed)
    started, done_now = time.time(), 0
    with open(labels_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        while queue and progress["photos"] < n_photos:
            pid, market, split, text = queue.popleft()
            if pid in tried:
                continue
            tried.add(pid)
            progress["tried"].append(pid)
            url = ingredient_photo(pid, session)
            time.sleep(API_PAUSE)
            if not url:
                progress["no_photo"] += 1
                continue
            r = get(url, session)
            if r is None or r.status_code != 200:
                progress["failed"] += 1
                continue
            photo = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
            if photo is None:
                progress["failed"] += 1
                continue
            h, w = photo.shape[:2]
            if max(h, w) > MAX_SIDE:
                f_ = MAX_SIDE / max(h, w)
                photo = cv2.resize(photo, None, fx=f_, fy=f_, interpolation=cv2.INTER_AREA)
            try:
                found = crops_of_photo(photo, clean_text(text))
            except Exception:                                      # one unreadable photo must not stop the run
                progress["failed"] += 1
                continue
            for k, (crop, label, reading, conf) in enumerate(found):
                name = f"{pid}_{k:03d}.jpg"
                cv2.imwrite(str(OUT_DIR / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                writer.writerow({"crop": name, "product_id": pid, "country_group": market, "split": split,
                                 "label": label, "rapidocr": reading, "confidence": round(conf, 4)})
            f.flush()
            progress["photos"] += 1
            progress["crops"] += len(found)
            progress["by_market"][market] = progress["by_market"].get(market, 0) + 1
            done_now += 1
            if progress["photos"] % 25 == 0:
                progress_path.write_text(json.dumps(progress))
                rate = (time.time() - started) / max(done_now, 1)
                print(f"{progress['photos']} photos, {progress['crops']} crops, {progress['no_photo']} without "
                      f"photo, {progress['failed']} failed | {rate:.1f} s/photo, "
                      f"~{rate * (n_photos - progress['photos']) / 3600:.1f} h left | {progress['by_market']}", flush=True)
    progress_path.write_text(json.dumps(progress))
    print(f"done: {progress['photos']} photos -> {progress['crops']} labelled crops", flush=True)
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photos", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    a = parser.parse_args()
    collect(a.photos, a.seed)


if __name__ == "__main__":
    main()
