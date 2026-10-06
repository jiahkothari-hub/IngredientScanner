"""Data validation: the pipeline must FAIL LOUDLY on malformed data instead of training on it.

    from src.utils.validation import validate_records
    validate_records(records, name="silver train")       # raises DataValidationError on problems

Checks (each produces a readable message with the record id):
  * empty text / no tokens
  * duplicate record ids
  * tokens / tags / offsets of different lengths
  * unknown tags (not in label2id)
  * impossible BIO sequences (I-X after O, or I-X after a different type)
  * offsets that do not match the text (text[start:end] != token)
  * entity text that does not match the text span
  * extremely long texts (more than MAX_TOKENS word tokens)
  * unexpected characters (control characters after normalisation)
  * inconsistent annotation: the same surface string labelled with different types
    (reported as a WARNING with counts - some inconsistency is legitimate, e.g. "colour")
"""
import unicodedata
from collections import Counter, defaultdict

from src.labeling.bio import LABEL2ID

MAX_TOKENS = 400


class DataValidationError(ValueError):
    pass


def check_bio_sequence(tags: list) -> list:
    """Return a list of problems with a BIO tag sequence (empty list = valid)."""
    problems, previous = [], "O"
    for i, tag in enumerate(tags):
        if tag not in LABEL2ID:
            problems.append(f"unknown tag {tag!r} at position {i}")
        elif tag.startswith("I-"):
            if previous == "O" or previous[2:] != tag[2:]:
                problems.append(f"{tag} at position {i} follows {previous}")
        previous = tag
    return problems


def validate_record(record: dict) -> list:
    rid = record.get("id", "?")
    problems = []
    text, tokens = record.get("text", ""), record.get("tokens", [])
    tags, offsets = record.get("ner_tags", []), record.get("token_offsets", [])
    if not text.strip() or not tokens:
        return [f"[{rid}] empty text or no tokens"]
    if not (len(tokens) == len(tags) == len(offsets)):
        problems.append(f"[{rid}] length mismatch: {len(tokens)} tokens, {len(tags)} tags, {len(offsets)} offsets")
    if len(tokens) > MAX_TOKENS:
        problems.append(f"[{rid}] extremely long: {len(tokens)} tokens (max {MAX_TOKENS})")
    for token, (start, end) in zip(tokens, offsets):
        if text[start:end] != token:
            problems.append(f"[{rid}] offset mismatch: {token!r} vs text {text[start:end]!r}")
            break
    problems += [f"[{rid}] {p}" for p in check_bio_sequence(tags)]
    for ent in record.get("entities", []):
        if text[ent["start"]:ent["end"]] != ent["text"]:
            problems.append(f"[{rid}] entity text {ent['text']!r} does not match span")
        if ent["label"] not in {t[2:] for t in LABEL2ID if t != "O"}:
            problems.append(f"[{rid}] invalid entity label {ent['label']!r}")
    bad_chars = {ch for ch in text if unicodedata.category(ch) in ("Cc", "Cf")}
    if bad_chars:
        problems.append(f"[{rid}] unexpected control characters {sorted(bad_chars)!r}")
    return problems


def label_consistency(records: list, min_count: int = 5) -> list:
    """Surface strings labelled with more than one type, e.g. ('colour', {'FUNCTION_CLASS': 90, 'COLOUR': 3})."""
    seen = defaultdict(Counter)
    for record in records:
        for ent in record.get("entities", []):
            seen[ent["text"].lower()][ent["label"]] += 1
    return sorted(((s, dict(c)) for s, c in seen.items() if len(c) > 1 and sum(c.values()) >= min_count),
                  key=lambda x: -sum(x[1].values()))


def validate_records(records: list, name: str = "dataset", raise_on_error: bool = True) -> dict:
    problems = []
    ids = Counter(r.get("id") for r in records)
    problems += [f"duplicate id {rid!r} ({n} times)" for rid, n in ids.items() if n > 1]
    for record in records:
        problems += validate_record(record)
    report = {"name": name, "records": len(records), "problems": problems,
              "inconsistent_surface_forms": label_consistency(records)[:25]}
    if problems and raise_on_error:
        shown = "\n  ".join(problems[:20])
        raise DataValidationError(f"{name}: {len(problems)} problem(s), first ones:\n  {shown}")
    return report
