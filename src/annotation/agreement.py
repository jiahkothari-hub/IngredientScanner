"""Inter-annotator agreement on the shared agreement items.

Usage:  python -m src.annotation.agreement       (after exporting each annotator's Doccano project)

For every pair of annotators we compute, on the products both annotated:
  * entity-level F1, treating one annotator as "gold" and the other as "prediction"
    (F1 is symmetric, so the choice does not matter) - overall and per class
  * Cohen's kappa on token labels (corrects for agreement by chance)
Low agreement on a class means the guideline for that class is unclear -> discuss, update
reports/entity_schema.md, and re-check those items BEFORE annotating the rest.

Output: reports/inter_annotator_agreement.json (+ printed summary)
"""
import itertools
import json

from src.annotation.import_annotations import EXPORT_DIR, load_exports
from src.labeling.bio import entities_to_bio
from src.evaluation.metrics import evaluate_entities, token_cohen_kappa
from src.preprocessing.pipeline import process_ingredient_text
from src.utils.config import project_path


def agreement(versions: dict) -> dict:
    annotators = sorted({a for by in versions.values() for a in by if a != "adjudicated"})
    results = {}
    for a, b in itertools.combinations(annotators, 2):
        shared = [pid for pid, by in versions.items() if a in by and b in by]
        if not shared:
            continue
        docs_a, docs_b, tags_a, tags_b = [], [], [], []
        for pid in shared:
            text = versions[pid][a][0]
            offsets = process_ingredient_text(text)["token_offsets"]
            ents_a = [{"start": s, "end": e, "label": l} for s, e, l in versions[pid][a][1]]
            ents_b = [{"start": s, "end": e, "label": l} for s, e, l in versions[pid][b][1]]
            docs_a.append(ents_a)
            docs_b.append(ents_b)
            try:
                tags_a.append(entities_to_bio(offsets, ents_a))
                tags_b.append(entities_to_bio(offsets, ents_b))
            except ValueError:
                pass          # misaligned spans are reported by import_annotations; skip for kappa
        scores = evaluate_entities(docs_a, docs_b)
        results[f"{a} vs {b}"] = {
            "documents": len(shared),
            "entity_f1": scores["micro"]["f1"],
            "per_class_f1": {k: v["f1"] for k, v in scores["per_class"].items() if v["support"] or v["predicted"]},
            "token_kappa": token_cohen_kappa(tags_a, tags_b),
        }
    return results


def main():
    results = agreement(load_exports(EXPORT_DIR))
    if not results:
        raise SystemExit("Need exports from at least two annotators with shared products.")
    project_path("reports/inter_annotator_agreement.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for pair, r in results.items():
        print(f"{pair}: {r['documents']} docs, entity F1 {r['entity_f1']:.3f}, token kappa {r['token_kappa']:.3f}")


if __name__ == "__main__":
    main()
