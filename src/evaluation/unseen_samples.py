"""Does the app work on NEW products? Two checks on samples that were not used to build anything.

Usage:  python -m src.evaluation.unseen_samples

1. Typed lists (data/test_samples/typed_products.json): 15 ingredient lists in the style of popular
   products, with the expected sugars / sweeteners / fats / preservatives / colours and additive numbers
   written by hand. Measures recall (expected items found) and precision of the category items and of
   the additive numbers.
2. New photos (data/images/unseen/): 16 real packet photos from Open Food Facts of products that are
   NOT among the 60 example photos (Nutella, Coca-Cola, KitKat, Maggi, Oreo, Haribo, Pringles ...).
   For each photo, the app's result is compared with the result on the product's typed ingredient list:
   how many of the sugars/fats/... and additive numbers found in the typed list are also found from
   the photo. This isolates what the photo (OCR) step loses.

Outputs: reports/unseen_samples.json and reports/unseen_samples.md
"""
import json
import re

import pandas as pd

from src.utils.config import project_path

CATEGORIES = ["SUGAR", "SWEETENER", "FAT", "PRESERVATIVE", "COLOUR"]


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def additive_numbers(result: dict) -> set:
    numbers = set()
    for e in result["entities"]:
        m = re.match(r"\d+[a-f]?", e.get("ins_number") or "")
        if m:
            numbers.add(m.group(0))
    return numbers


def items_found(expected: list, found: list) -> int:
    """An expected item counts as found if it appears in (or contains) one of the found item names."""
    found_norm = [norm(f) for f in found]
    return sum(any(norm(x) in f or f in norm(x) for f in found_norm if f) for x in expected)


def evaluate_typed(scanner) -> dict:
    data = json.loads(project_path("data/test_samples/typed_products.json").read_text(encoding="utf-8"))
    rows, totals = [], {"cat_expected": 0, "cat_found": 0, "cat_predicted": 0, "cat_correct": 0,
                        "add_expected": 0, "add_found": 0, "add_predicted": 0}
    for p in data["products"]:
        result = scanner.scan_text(p["ingredients_text"])
        groups = result["summary"]["groups"]
        expected_items = sum(len(v) for v in p["expected"].values())
        found = sum(items_found(p["expected"].get(c, []), groups[c]) for c in CATEGORIES)
        predicted = sum(len(groups[c]) for c in CATEGORIES)
        correct = sum(sum(any(norm(x) in norm(g) or norm(g) in norm(x) for x in p["expected"].get(c, []))
                          for g in groups[c]) for c in CATEGORIES)
        numbers, expected_numbers = additive_numbers(result), set(p["expected_additive_numbers"])
        totals["cat_expected"] += expected_items
        totals["cat_found"] += found
        totals["cat_predicted"] += predicted
        totals["cat_correct"] += correct
        totals["add_expected"] += len(expected_numbers)
        totals["add_found"] += len(numbers & expected_numbers)
        totals["add_predicted"] += len(numbers)
        rows.append({"product": p["product"], "category items found": f"{found}/{expected_items}",
                     "extra category items": predicted - correct,
                     "additive numbers found": f"{len(numbers & expected_numbers)}/{len(expected_numbers)}",
                     "missed": ", ".join(sorted(expected_numbers - numbers)) or "-"})
    summary = {"category_recall": round(totals["cat_found"] / totals["cat_expected"], 3),
               "category_precision": round(totals["cat_correct"] / max(1, totals["cat_predicted"]), 3),
               "additive_recall": round(totals["add_found"] / totals["add_expected"], 3),
               "additive_precision": round(totals["add_found"] / max(1, totals["add_predicted"]), 3)}
    return {"summary": summary, "rows": rows}


def evaluate_photos(scanner) -> dict:
    index = pd.read_csv(project_path("data/images/unseen/unseen.csv"), dtype={"product_id": str})
    rows, cat_ref = [], [0, 0]
    add = [0, 0, 0]          # found from photo & in typed, in typed, found from photo
    for r in index.itertuples():
        photo = scanner.scan_image(project_path(r.image))
        typed = scanner.scan_text(r.reference_text)
        ref_items = {c: typed["summary"]["groups"][c] for c in CATEGORIES}
        n_ref = sum(len(v) for v in ref_items.values())
        n_found = sum(items_found(ref_items[c], photo["summary"]["groups"][c]) for c in CATEGORIES)
        p_num, t_num = additive_numbers(photo), additive_numbers(typed)
        cat_ref[0] += n_found
        cat_ref[1] += n_ref
        add[0] += len(p_num & t_num)
        add[1] += len(t_num)
        add[2] += len(p_num)
        rows.append({"product": f"{r.product_name} ({r.brands})", "rotation": photo["ocr"]["rotation"],
                     "OCR confidence": photo["ocr"]["mean_confidence"], "section": photo["ocr"]["section_method"],
                     "category items found": f"{n_found}/{n_ref}",
                     "additive numbers found": f"{len(p_num & t_num)}/{len(t_num)}",
                     "seconds": photo["ocr"]["seconds"]})
    summary = {"photos": len(index),
               "category_items_recovered": round(cat_ref[0] / max(1, cat_ref[1]), 3),
               "additive_numbers_recovered": round(add[0] / max(1, add[1]), 3),
               "additive_numbers_precision": round(add[0] / max(1, add[2]), 3)}
    return {"summary": summary, "rows": rows}


def main():
    from src.app.scanner import IngredientScanner
    scanner = IngredientScanner("auto")
    typed = evaluate_typed(scanner)
    photos = evaluate_photos(scanner)
    report = {"typed_lists": typed, "new_photos": photos}
    project_path("reports/unseen_samples.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    md = ["# Unseen samples: typed lists and new photos", "",
          "Produced by `python -m src.evaluation.unseen_samples` (classifier: hybrid). See the module docstring.", "",
          "## 1. Typed ingredient lists (15 products, hand-written expectations)", "",
          "| Metric | Value |", "|---|---:|"]
    md += [f"| {k.replace('_', ' ')} | {v} |" for k, v in typed["summary"].items()]
    md += ["", pd.DataFrame(typed["rows"]).to_markdown(index=False), "",
           "## 2. New real photos (16 products not among the example photos)", "",
           "Reference = the app's own result on the product's typed ingredient list, so this measures what the "
           "photo step loses.", "", "| Metric | Value |", "|---|---:|"]
    md += [f"| {k.replace('_', ' ')} | {v} |" for k, v in photos["summary"].items()]
    md += ["", pd.DataFrame(photos["rows"]).to_markdown(index=False), ""]
    project_path("reports/unseen_samples.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"typed": typed["summary"], "photos": photos["summary"]}, indent=1))


if __name__ == "__main__":
    main()
