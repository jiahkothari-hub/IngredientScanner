"""Collect real packet photos (ingredient labels) for the OCR pipeline and its evaluation.

Usage:  python -m src.ocr.collect_images [--per-group 15]

Source: Open Food Facts' public image bucket on AWS (openfoodfacts-images, CC BY-SA licence).
Each product folder contains the raw photos (1.jpg, 2.jpg, ...) and, for each photo, the OCR text
that Open Food Facts stored for it (1.json.gz ...). Photos are not tagged as "ingredients" or "front",
so for every product we pick the photo whose stored OCR text contains the most words of the product's
known ingredient list. That photo is almost always the ingredients panel.

Products come from the silver TEST split only, so the end-to-end evaluation never uses products the
NER models were trained on. India is over-sampled because Indian labels are the main target style.

Outputs:
  data/images/packets/<product_id>.jpg   resized to max 1600 px (keeps the repository small)
  data/images/packets.csv                product_id, market, image, match score, reference text
"""
import argparse
import gzip
import io
import json
import random
import re

import pandas as pd
import requests
from PIL import Image

from src.ner.data import load_split
from src.utils.config import project_path

BUCKET = "https://openfoodfacts-images.s3.eu-west-3.amazonaws.com"
OUT_DIR = project_path("data/images/packets")
INDEX_CSV = project_path("data/images/packets.csv")
MAX_SIDE = 1600
MIN_MATCH = 0.6
GROUP_WEIGHTS = {"india": 2, "uk_ireland": 1, "north_america": 1, "other_english": 1}
WORD_RE = re.compile(r"[a-z]{3,}")


def product_folders(code: str) -> list:
    """Bucket folder(s) for a barcode: 8901058000290 -> data/890/105/800/0290.
    Short codes may be stored zero-padded to 13 digits, so both spellings are tried."""
    candidates = [code, code.zfill(13)] if len(code) < 13 else [code]
    folders = []
    for c in dict.fromkeys(candidates):
        m = re.match(r"^(\d{3})(\d{3})(\d{3})(\d+)$", c)
        folders.append(f"data/{m.group(1)}/{m.group(2)}/{m.group(3)}/{m.group(4)}" if m else f"data/{c}")
    return folders


def list_images(folder: str) -> list:
    """Image ids that have both a photo and stored OCR text."""
    xml = requests.get(f"{BUCKET}/?list-type=2&max-keys=200&prefix={folder}/", timeout=30).text
    keys = re.findall(r"<Key>([^<]+)</Key>", xml)
    photos = {m.group(1) for k in keys if (m := re.search(r"/(\d+)\.jpg$", k))}
    ocr = {m.group(1) for k in keys if (m := re.search(r"/(\d+)\.json\.gz$", k))}
    return sorted(photos & ocr, key=int)


def stored_ocr_text(folder: str, image_id: str) -> str:
    raw = requests.get(f"{BUCKET}/{folder}/{image_id}.json.gz", timeout=30).content
    data = json.loads(gzip.decompress(raw))
    responses = data.get("responses", [data])
    return responses[0].get("fullTextAnnotation", {}).get("text", "") if responses else ""


def match_score(reference: str, ocr_text: str) -> float:
    """Share of the ingredient list's words that also appear in the photo's OCR text."""
    ref = set(WORD_RE.findall(reference.lower()))
    return len(ref & set(WORD_RE.findall(ocr_text.lower()))) / len(ref) if ref else 0.0


def find_ingredient_photo(code: str, reference: str):
    best = None
    for folder in product_folders(code):
        try:
            ids = list_images(folder)
        except requests.RequestException:
            continue
        for image_id in ids[:12]:
            try:
                score = match_score(reference, stored_ocr_text(folder, image_id))
            except (requests.RequestException, ValueError, OSError):
                continue
            if best is None or score > best[2]:
                best = (folder, image_id, score)
        if ids:
            break
    return best


def download_resized(folder: str, image_id: str, path):
    raw = requests.get(f"{BUCKET}/{folder}/{image_id}.jpg", timeout=60).content
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    image.save(path, "JPEG", quality=85)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-group", type=int, default=12, help="photos per market (India gets twice as many)")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test = load_split("silver", "test")
    rng = random.Random(7)
    rows = []
    for group, weight in GROUP_WEIGHTS.items():
        # medium-length lists: long enough to be interesting, short enough to fit on one photo
        pool = [r for r in test if r["country_group"] == group and 15 <= len(r["tokens"]) <= 150]
        rng.shuffle(pool)
        target, found = args.per_group * weight, 0
        for r in pool:
            if found >= target:
                break
            best = find_ingredient_photo(r["product_id"], r["text"])
            if not best or best[2] < MIN_MATCH:
                continue
            folder, image_id, score = best
            path = OUT_DIR / f"{r['product_id']}.jpg"
            download_resized(folder, image_id, path)
            rows.append({"product_id": r["product_id"], "country_group": group,
                         "image": str(path.relative_to(project_path(""))), "source_image": f"{folder}/{image_id}.jpg",
                         "match_score": round(score, 3), "reference_text": r["text"]})
            found += 1
            print(f"  {group}: {found}/{target}  {r['product_id']}  match {score:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(INDEX_CSV, index=False)
    print(f"Saved {len(rows)} photos -> {INDEX_CSV.relative_to(project_path(''))}")


if __name__ == "__main__":
    main()
