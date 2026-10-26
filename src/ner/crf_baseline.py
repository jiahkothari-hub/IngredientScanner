"""Conditional Random Field (CRF) NER baseline.

Usage:  python -m src.ner.crf_baseline          (trains on silver train, saves models/crf/crf_model.crfsuite)

What a CRF is: a classic statistical sequence model. Every word is described by hand-made features
(the word itself, its suffix, whether it is a number, the neighbouring words ...). The CRF learns a
weight for every feature-label combination AND for every label-label transition (e.g. I-SUGAR may
follow B-SUGAR but not O), and finds the best tag sequence for a whole sentence at once.

Why it is here: it sits between the dictionary rules and the Transformers. It learns from data like
the Transformers, but it has no pretrained knowledge of English, so it shows how much the pretrained
language model adds. We deliberately give it NO dictionary features, so it has to learn the labels
from the context words themselves.
"""
import json
import re
import time

import sklearn_crfsuite

from src.ner.data import load_split, tags_of
from src.utils.config import project_path

MODEL_DIR = project_path("models/crf")
MODEL_PATH = MODEL_DIR / "crf_model.crfsuite"


def word_shape(word: str) -> str:
    """'INS' -> 'XX', 'E330' -> 'Xdd', 'Sugar' -> 'Xxx' (X upper, x lower, d digit; runs cut to 2)."""
    shape = "".join("X" if ch.isupper() else "x" if ch.islower() else "d" if ch.isdigit() else ch for ch in word)
    return re.sub(r"(.)\1{2,}", r"\1\1", shape)


def word_features(tokens: list, i: int) -> dict:
    word = tokens[i]
    lower = word.lower()
    features = {
        "bias": 1.0,
        "lower": lower,
        "suffix3": lower[-3:],
        "suffix2": lower[-2:],
        "prefix3": lower[:3],
        "shape": word_shape(word),
        "is_digit": word.isdigit(),
        "is_title": word.istitle(),
        "is_upper": word.isupper(),
        "is_punct": not any(ch.isalnum() for ch in word),
        "has_digit": any(ch.isdigit() for ch in word),
        "length": min(len(word), 10),
    }
    for offset in (-2, -1, 1, 2):                   # context window of two words on each side
        j = i + offset
        if 0 <= j < len(tokens):
            features[f"{offset}:lower"] = tokens[j].lower()
            features[f"{offset}:shape"] = word_shape(tokens[j])
        else:
            features[f"{offset}:edge"] = True
    return features


def sentence_features(tokens: list) -> list:
    return [word_features(tokens, i) for i in range(len(tokens))]


class CRFTagger:
    name = "crf"

    def __init__(self, model_path=MODEL_PATH):
        self.crf = sklearn_crfsuite.CRF(model_filename=str(model_path))

    def predict_tags(self, tokens: list) -> list:
        return self.crf.predict_single(sentence_features(tokens)) if tokens else []

    def predict_records(self, records: list) -> list:
        return self.crf.predict([sentence_features(r["tokens"]) for r in records])


TRAIN_LIMIT = 8000   # the same seeded 8,000-sentence sample as the Transformer runs (fair comparison)


def main():
    train = load_split("silver", "train", limit=TRAIN_LIMIT)
    print(f"Training CRF on {len(train):,} silver sentences ...", flush=True)
    start = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.05, c2=0.01, max_iterations=150,
                               all_possible_transitions=True, model_filename=str(MODEL_PATH))
    crf.fit([sentence_features(r["tokens"]) for r in train], [tags_of(r) for r in train])
    info = {"train_sentences": len(train), "seconds": round(time.time() - start),
            "c1": 0.05, "c2": 0.01, "max_iterations": 150, "features": "word, affixes, shape, +-2 context words"}
    (MODEL_DIR / "training_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Saved {MODEL_PATH.relative_to(project_path(''))} in {info['seconds']}s")


if __name__ == "__main__":
    main()
