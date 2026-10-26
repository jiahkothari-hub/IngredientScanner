"""End-to-end evaluation on real packet photos: photo -> OCR -> section -> NER, vs the known text.

Usage:
    python -m src.ocr.evaluate_ocr --run-ocr        # OCR all photos once (slow on CPU), cached
    python -m src.ocr.evaluate_ocr                  # evaluate OCR + every available NER system

Reference: for each photo we know the product's ingredient list (typed into Open Food Facts) and its
silver entities. These products are in the held-out TEST split, so no model was trained on them.

Measured:
  OCR quality        word recall of the FULL OCR text (can the OCR read the list at all?), word recall
                     and character error rate (CER) of the extracted section, how the section was found.
                     Per-photo values: reports/ocr_per_photo.csv
  Entity agreement   entities found in the OCR text vs the reference entities, compared as
                     (label, normalised text). Offsets cannot be compared because OCR text differs.
                     "fuzzy" also accepts small spelling differences (similarity >= 85/100, same label).
  Additive recovery  additive numbers found through entity linking vs the reference's numbers -
                     the question the app must answer: "which additives are in this product?"
Caveat: the reference entities are silver (rule-made); the text reference is crowd-sourced.
"""
import argparse
import json
import re
from collections import Counter

import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from src.kb.entity_linking import get_linker
from src.labeling.lexicon import lookup_key
from src.ner.data import load_split
from src.utils.config import project_path
from src.utils.io import read_jsonl

INDEX = project_path("data/images/packets.csv")
OCR_CACHE = project_path("data/images/ocr_results.jsonl")


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def run_ocr():
    """OCR every photo once. Results are appended photo by photo, so an interrupted run resumes."""
    import json as _json
    from src.ocr.ocr_cleanup import fix_additive_codes
    from src.ocr.ocr_pipeline import read_image
    from src.ocr.section_extraction import extract_ingredients_section
    done = {r["product_id"] for r in read_jsonl(OCR_CACHE)} if OCR_CACHE.exists() else set()
    for photo in pd.read_csv(INDEX, dtype={"product_id": str}).itertuples():
        if photo.product_id in done:
            continue
        ocr = read_image(project_path(photo.image))
        cleaned = fix_additive_codes(ocr["text"])
        section = extract_ingredients_section(cleaned)
        row = {"product_id": photo.product_id, "country_group": photo.country_group,
               "ocr_text": ocr["text"], "corrected_text": cleaned, "section": section["text"],
               "section_method": section["method"], "rotation": ocr["rotation"],
               "mean_confidence": ocr["mean_confidence"]}
        with open(OCR_CACHE, "a", encoding="utf-8") as f:
            f.write(_json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {photo.product_id}: {section['method']}, {len(section['text'])} chars", flush=True)


def entity_keys(entities) -> list:
    return [(e["label"], re.sub(r"[^a-z0-9]", "", lookup_key(e["text"]))) for e in entities]


def bag_scores(pred: list, gold: list, fuzzy: bool) -> tuple:
    """Return (correct, n_pred, n_gold) for two bags of (label, key)."""
    remaining = list(gold)
    correct = 0
    for label, key in pred:
        for i, (gl, gk) in enumerate(remaining):
            if gl == label and (gk == key or (fuzzy and fuzz.ratio(gk, key) >= 85)):
                correct += 1
                remaining.pop(i)
                break
    return correct, len(pred), len(gold)


def prf(c, p, g) -> dict:
    precision, recall = (c / p if p else 0.0), (c / g if g else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def additive_numbers(entities) -> set:
    linker = get_linker()
    numbers = set()
    for e in entities:
        if e["label"] in ("INS_CODE", "ADDITIVE", "PRESERVATIVE", "COLOUR", "SWEETENER"):
            number = linker.link(e)["ins_number"]
            if number:
                numbers.add(re.match(r"\d+", number).group(0))
    return numbers


def load_systems() -> dict:
    from src.baseline.dictionary_ner import DictionaryNER
    systems = {"dictionary": DictionaryNER()}
    try:
        from src.ner.crf_baseline import MODEL_PATH, CRFTagger
        if MODEL_PATH.exists():
            systems["crf"] = CRFTagger()
    except ImportError:
        pass
    from src.ner.transformer_ner import TransformerTagger
    for run in sorted(project_path("models/runs").glob("*/best/config.json")):
        systems[run.parent.parent.name] = TransformerTagger(run.parent)
    from src.app.scanner import FINAL_MODEL_DIR
    from src.ner.hybrid import HybridNER
    final = TransformerTagger(FINAL_MODEL_DIR) if (FINAL_MODEL_DIR / "config.json").exists() else None
    systems["hybrid (app)"] = HybridNER(final)
    return systems


def predict_entities(system, text):
    if hasattr(system, "predict"):
        return system.predict(text)["entities"]
    from src.labeling.bio import bio_to_entities
    from src.preprocessing.pipeline import process_ingredient_text
    processed = process_ingredient_text(text)
    tags = system.predict_tags(processed["tokens"])
    return bio_to_entities(processed["token_offsets"], tags, processed["text"])


def word_recall(reference: str, hypothesis: str) -> float:
    ref_words = set(re.findall(r"[a-z]{3,}", reference))
    return len(ref_words & set(re.findall(r"[a-z]{3,}", hypothesis))) / max(1, len(ref_words))


def evaluate():
    from src.ocr.section_extraction import extract_ingredients_section
    ocr = {r["product_id"]: r for r in read_jsonl(OCR_CACHE)}
    for r in ocr.values():                 # re-run section extraction: it is cheap, OCR is not
        section = extract_ingredients_section(r["corrected_text"])
        r["section"], r["section_method"] = section["text"], section["method"]
    test = {r["id"]: r for r in load_split("silver", "test") if r["id"] in ocr}
    report = {"photos": len(ocr)}

    rows = []
    for pid, r in ocr.items():
        ref = normalise(test[pid]["text"])
        rows.append({"product_id": pid, "market": r["country_group"], "section_method": r["section_method"],
                     "cer_section": Levenshtein.normalized_distance(ref, normalise(r["section"])),
                     "word_recall_full_ocr": word_recall(ref, normalise(r["ocr_text"])),
                     "word_recall_section": word_recall(ref, normalise(r["section"])),
                     "ocr_confidence": r["mean_confidence"]})
    per_photo = pd.DataFrame(rows)
    per_photo.round(3).to_csv(project_path("reports/ocr_per_photo.csv"), index=False)
    report["ocr"] = {
        "median_cer_section": round(float(per_photo["cer_section"].median()), 3),
        "median_word_recall_full_ocr": round(float(per_photo["word_recall_full_ocr"].median()), 3),
        "median_word_recall_section": round(float(per_photo["word_recall_section"].median()), 3),
        "photos_readable (full-text word recall >= 0.5)": int((per_photo["word_recall_full_ocr"] >= 0.5).sum()),
        "section_methods": dict(Counter(per_photo["section_method"])),
        "by_market_median_word_recall_full_ocr": per_photo.groupby("market")["word_recall_full_ocr"].median().round(3).to_dict(),
    }

    readable = set(per_photo.loc[per_photo["word_recall_full_ocr"] >= 0.5, "product_id"])
    report["systems"], report["systems_readable_photos_only"] = {}, {}
    for name, system in load_systems().items():
        for subset_name, subset in (("systems", set(ocr)), ("systems_readable_photos_only", readable)):
            exact, fuzzy, add = [0, 0, 0], [0, 0, 0], [0, 0, 0]
            for pid in subset:
                r = ocr[pid]
                gold = entity_keys(test[pid]["entities"])
                pred_entities = predict_entities(system, r["section"])
                pred = entity_keys(pred_entities)
                for acc, scores in ((exact, bag_scores(pred, gold, False)), (fuzzy, bag_scores(pred, gold, True))):
                    for i in range(3):
                        acc[i] += scores[i]
                ours, ref = additive_numbers(pred_entities), additive_numbers(test[pid]["entities"])
                add[0] += len(ours & ref)
                add[1] += len(ours)
                add[2] += len(ref)
            report[subset_name][name] = {"photos": len(subset), "entities_exact": prf(*exact),
                                         "entities_fuzzy": prf(*fuzzy), "additive_numbers": prf(*add)}
        print(name, "all:", report["systems"][name]["additive_numbers"],
              "readable:", report["systems_readable_photos_only"][name]["entities_fuzzy"], flush=True)
    project_path("reports/ocr_end_to_end.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["ocr"], indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-ocr", action="store_true")
    args = parser.parse_args()
    if args.run_ocr or not OCR_CACHE.exists():
        run_ocr()           # skips photos that are already cached
    evaluate()


if __name__ == "__main__":
    main()
