"""Entity-level evaluation: Precision, Recall, F1 (overall and per class).

ONE scorer for every system (dictionary baseline, DistilBERT, BERT, BERT+CRF, inter-annotator
agreement), so all numbers in the report are comparable.

Entity-level, strict ("exact match"): a predicted entity is correct only if its start, end AND label
are all identical to a gold entity. This is the CoNLL / seqeval standard and is stricter than
token accuracy, where the many "O" tokens make every model look good.

    precision = correct / predicted      "of what the system found, how much is right?"
    recall    = correct / gold           "of what is really there, how much was found?"
    F1        = 2PR / (P + R)            harmonic mean

We also report a lenient "partial" score (same label, spans overlap) to separate boundary errors
from completely missed entities.

Person 2 can score token-level predictions directly:
    from src.evaluation.metrics import evaluate_tag_sequences
    evaluate_tag_sequences(gold_tag_lists, predicted_tag_lists)
"""
from collections import Counter

from src.labeling.bio import LABELS, bio_to_entities


def _prf(tp, n_pred, n_gold) -> dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "support": n_gold, "predicted": n_pred, "correct": tp}


def evaluate_entities(gold_docs: list, pred_docs: list, labels=LABELS) -> dict:
    """gold_docs / pred_docs: one list of entities per document; entities need start, end, label.

    Returns {"micro": {...}, "macro_f1": x, "per_class": {label: {...}}, "partial_micro": {...}}.
    """
    if len(gold_docs) != len(pred_docs):
        raise ValueError("gold and predictions must have the same number of documents")
    tp, n_gold, n_pred = Counter(), Counter(), Counter()
    partial_tp = 0
    for gold, pred in zip(gold_docs, pred_docs):
        gold_set = {(e["start"], e["end"], e["label"]) for e in gold}
        pred_set = {(e["start"], e["end"], e["label"]) for e in pred}
        for _, _, label in gold_set:
            n_gold[label] += 1
        for _, _, label in pred_set:
            n_pred[label] += 1
        for _, _, label in gold_set & pred_set:
            tp[label] += 1
        # partial: a prediction counts if it overlaps a gold entity with the same label (each gold used once)
        used = set()
        for ps, pe, pl in pred_set:
            for g in gold_set:
                if g not in used and g[2] == pl and ps < g[1] and g[0] < pe:
                    used.add(g)
                    partial_tp += 1
                    break
    per_class = {label: _prf(tp[label], n_pred[label], n_gold[label]) for label in labels}
    present = [label for label in labels if n_gold[label] > 0]
    return {
        "micro": _prf(sum(tp.values()), sum(n_pred.values()), sum(n_gold.values())),
        "macro_f1": round(sum(per_class[l]["f1"] for l in present) / len(present), 4) if present else 0.0,
        "per_class": per_class,
        "partial_micro": _prf(partial_tp, sum(n_pred.values()), sum(n_gold.values())),
        "documents": len(gold_docs),
    }


def evaluate_tag_sequences(gold_tags: list, pred_tags: list) -> dict:
    """Score BIO tag sequences (list of lists of tag strings) at entity level.
    Token positions are used as offsets, so no text is needed."""
    gold_docs, pred_docs = [], []
    for g, p in zip(gold_tags, pred_tags):
        if len(g) != len(p):
            raise ValueError("gold and predicted sequences differ in length")
        offsets = [(i, i + 1) for i in range(len(g))]
        gold_docs.append(bio_to_entities(offsets, g))
        pred_docs.append(bio_to_entities(offsets, p))
    return evaluate_entities(gold_docs, pred_docs)


def token_cohen_kappa(tags_a: list, tags_b: list) -> float:
    """Cohen's kappa between two annotators on token labels (entity type, B/I ignored).
    kappa = (observed agreement - chance agreement) / (1 - chance agreement)."""
    a = [t.split("-", 1)[-1] for seq in tags_a for t in seq]
    b = [t.split("-", 1)[-1] for seq in tags_b for t in seq]
    n = len(a)
    if n == 0:
        return 0.0
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in ca) / (n * n)
    return round((observed - expected) / (1 - expected), 4) if expected < 1 else 1.0


def format_report(result: dict, title: str) -> str:
    """Markdown table of a result from evaluate_entities()."""
    lines = [f"### {title}", "", "| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    for label, m in result["per_class"].items():
        if m["support"] or m["predicted"]:
            lines.append(f"| {label} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |")
    m = result["micro"]
    lines.append(f"| **overall (micro)** | **{m['precision']:.3f}** | **{m['recall']:.3f}** | **{m['f1']:.3f}** | {m['support']} |")
    lines.append(f"| macro F1 | | | {result['macro_f1']:.3f} | |")
    pm = result["partial_micro"]
    lines.append(f"| partial match (micro) | {pm['precision']:.3f} | {pm['recall']:.3f} | {pm['f1']:.3f} | |")
    return "\n".join(lines)
