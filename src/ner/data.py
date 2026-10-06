"""Shared data helpers for all NER systems (CRF, DistilBERT, BERT, dictionary baseline).

Every system is evaluated on the same records with the same scorer. The evaluation sets are:

  silver_test        held-out products, labels from the rules. Measures how well a model learned the
                     labelling policy on unseen products (but the rules are the reference).
  noisy_test_<rate>  the same products with OCR-style character confusions (l<->1, O<->0, ...).
                     The labels are those of the CLEAN text. This measures robustness to OCR, which
                     rules cannot learn. Only letter/digit substitutions are used, so word boundaries,
                     tokens and therefore labels stay aligned.
  gold_test          human-verified (data/splits/gold/test.jsonl), the final benchmark, once annotated.
"""
import random

from src.evaluation.noise_robustness import _SUBSTITUTE
from src.utils.config import project_path
from src.utils.io import read_jsonl


def load_split(source: str = "silver", split: str = "train", limit: int = None, seed: int = 42) -> list:
    suffix = ".jsonl.gz" if source == "silver" else ".jsonl"
    path = project_path(f"data/splits/{source}/{split}{suffix}")
    if not path.exists():
        return []
    records = read_jsonl(path)
    if limit and len(records) > limit:
        records = random.Random(seed).sample(records, limit)
    return records


def noisy_copy(records: list, rate: float, seed: int = 0) -> list:
    """Copy records with OCR-style noise in the text; tokens are re-read from the same offsets.

    Characters inside percentage tokens ("5%") are not changed: "S%" would be split into two words
    by the tokenizer and the labels would no longer line up. Everything else, including additive
    codes ("330" -> "33O"), can receive noise.
    """
    rng = random.Random(seed)
    out = []
    for r in records:
        protected = {i for (s, e), tok in zip(r["token_offsets"], r["tokens"]) if tok.endswith("%")
                     for i in range(s, e)}
        chars = list(r["text"])
        for i, ch in enumerate(chars):
            if i not in protected and ch in _SUBSTITUTE and rng.random() < rate:
                chars[i] = rng.choice(_SUBSTITUTE[ch])
        text = "".join(chars)
        tokens = [text[s:e] for s, e in r["token_offsets"]]
        entities = [{**e, "text": text[e["start"]:e["end"]]} for e in r["entities"]]
        out.append({**r, "text": text, "tokens": tokens, "entities": entities})
    return out


def tags_of(record: dict) -> list:
    """BIO tag names of a split record (split files store integer ids plus names)."""
    return record.get("ner_tag_names") or record["ner_tags"]


def evaluation_sets(limit: int = None) -> dict:
    """All evaluation sets that exist right now (gold appears automatically after annotation)."""
    silver_test = load_split("silver", "test", limit=limit)
    sets = {"silver_test": silver_test,
            "noisy_test_0.05": noisy_copy(silver_test, 0.05, seed=1),
            "noisy_test_0.10": noisy_copy(silver_test, 0.10, seed=2)}
    gold = load_split("gold", "test")
    if gold:
        sets["gold_test"] = gold
    return sets
