"""Import human-verified annotations from Doccano and build the GOLD dataset.

Usage:
  1. In Doccano, export each annotator's project as JSONL with "Export only approved documents"
     ticked, and save it as  data/annotations/exports/<annotator>.jsonl
     (approved = the annotator has checked every entity of that document).
  2. If agreement items were annotated differently, the team agrees on one version and saves it in
     data/annotations/exports/adjudicated.jsonl (same format).
  3. python -m src.annotation.import_annotations

Output:
  data/annotations/gold_verified.jsonl            GOLD records (same format as silver, label_source="gold_human")
  data/splits/gold/{train,validation,test}.jsonl  gold split files for Person 2
  reports/gold_import_report.json                 counts, warnings, unresolved disagreements

Only documents that a person exported as approved become gold. Nothing is gold by default.
"""
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.labeling.bio import LABELS, entities_to_bio
from src.preprocessing.pipeline import process_ingredient_text
from src.utils.config import project_path
from src.utils.io import read_jsonl, write_jsonl
from src.utils.validation import validate_records

EXPORT_DIR = project_path("data/annotations/exports")
GOLD_PATH = project_path("data/annotations/gold_verified.jsonl")


def parse_doccano_line(line: dict) -> list:
    """Return [(start, end, label)] from either Doccano export format:
       {"label": [[0, 5, "SUGAR"], ...]}   or   {"entities": [{"start_offset":0,"end_offset":5,"label":"SUGAR"}]}"""
    if "entities" in line:
        return [(e["start_offset"], e["end_offset"], e["label"]) for e in line["entities"]]
    return [tuple(x) for x in line.get("label", [])]


def snap_to_tokens(text, token_offsets, start, end, warnings, pid):
    """Trim spaces/punctuation the annotator included and expand spans that cut through a token."""
    while start < end and not text[start].isalnum():
        start += 1
    while end > start and not text[end - 1].isalnum() and text[end - 1] not in ")]":
        end -= 1
    covering = [(s, e) for s, e in token_offsets if s < end and e > start]
    if not covering:
        return None
    new_start, new_end = covering[0][0], covering[-1][1]
    if (new_start, new_end) != (start, end):
        warnings.append(f"[{pid}] span {text[start:end]!r} expanded to token boundaries {text[new_start:new_end]!r}")
    return new_start, new_end


def to_gold_record(pid, text, spans, annotators, product_meta, warnings) -> dict:
    processed = process_ingredient_text(text)
    if processed["text"] != text:
        raise ValueError(f"[{pid}] annotated text differs from the normalised text - was it edited?")
    entities = []
    for start, end, label in sorted(spans):
        if label not in LABELS:
            raise ValueError(f"[{pid}] unknown label {label!r}")
        snapped = snap_to_tokens(text, processed["token_offsets"], start, end, warnings, pid)
        if snapped:
            entities.append({"start": snapped[0], "end": snapped[1], "text": text[snapped[0]:snapped[1]],
                             "label": label})
    return {
        "id": pid,
        "product_id": pid,
        "country_group": product_meta.get("country_group", ""),
        "text_raw": product_meta.get("text_raw", text),
        "text": text,
        "tokens": processed["tokens"],
        "token_offsets": processed["token_offsets"],
        "ner_tags": entities_to_bio(processed["token_offsets"], entities),
        "entities": entities,
        "label_source": "gold_human",
        "annotators": annotators,
        "status": "verified",
    }


def load_exports(export_dir: Path) -> dict:
    """product_id -> {annotator: (text, spans)}; 'adjudicated' is treated as a special annotator."""
    versions = defaultdict(dict)
    for path in sorted(export_dir.glob("*.jsonl")):
        for line in read_jsonl(path):
            pid = str(line.get("product_id") or line.get("meta", {}).get("product_id"))
            versions[pid][path.stem] = (line["text"], sorted(parse_doccano_line(line)))
    return versions


def build_gold(export_dir=EXPORT_DIR, silver_path=project_path("data/processed/silver.jsonl.gz")):
    silver = {r["id"]: r for r in read_jsonl(silver_path)}
    versions = load_exports(export_dir)
    records, warnings, unresolved = [], [], []
    for pid, by_annotator in sorted(versions.items()):
        meta = silver.get(pid, {})
        if "adjudicated" in by_annotator:
            text, spans = by_annotator["adjudicated"]
            annotators = sorted(a for a in by_annotator if a != "adjudicated") + ["adjudicated"]
        else:
            distinct = {tuple(spans) for _, spans in by_annotator.values()}
            if len(distinct) > 1:
                unresolved.append(pid)          # annotators disagree and nobody adjudicated yet
                continue
            text, spans = next(iter(by_annotator.values()))
            annotators = sorted(by_annotator)
        records.append(to_gold_record(pid, text, spans, annotators, meta, warnings))
    return records, warnings, unresolved


def main():
    if not EXPORT_DIR.exists() or not list(EXPORT_DIR.glob("*.jsonl")):
        raise SystemExit(f"No Doccano exports found in {EXPORT_DIR.relative_to(project_path(''))}/ - "
                         "annotate the batches first (see README section 7).")
    records, warnings, unresolved = build_gold()
    validate_records(records, name="gold")
    write_jsonl(records, GOLD_PATH)

    # gold split files use the SAME split assignment as silver (no leakage between them)
    from src.data.split_dataset import write_split_files
    assignment = pd.read_csv(project_path("data/processed/split_assignment.csv"), dtype={"product_id": str})
    split_of = dict(zip(assignment["product_id"], assignment["split"]))
    counts = write_split_files(records, split_of, "gold", compress=False)

    report = {"gold_records": len(records), "gold_split_sizes": counts,
              "unresolved_disagreements": unresolved, "span_warnings": warnings[:200]}
    project_path("reports/gold_import_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Gold: {len(records)} verified records {counts}; {len(unresolved)} unresolved; "
          f"{len(warnings)} span warnings")


if __name__ == "__main__":
    main()
