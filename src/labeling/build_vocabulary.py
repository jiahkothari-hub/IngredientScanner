"""Build the corpus vocabulary used by the weak labeller's fallback rule.

Usage:  python -m src.labeling.build_vocabulary

Why: the fallback rule ("a short leftover piece is an INGREDIENT") also fires on OCR garbage
("acd anin b-12"), addresses and phone numbers that Open Food Facts texts contain. Real ingredient
words occur in many different products; garbage words almost never repeat. So a fallback entity is
only created when every word (3+ letters) occurs in at least MIN_DOC_FREQ products.

Output: data/processed/ingredient_vocabulary.txt  (one word per line, with its document frequency)
"""
import re
from collections import Counter

import pandas as pd

from src.preprocessing.normalize import normalize_text
from src.utils.config import load_config, project_path

MIN_DOC_FREQ = 3
VOCAB_PATH = project_path("data/processed/ingredient_vocabulary.txt")
WORD_RE = re.compile(r"[^\W\d_]{3,}")


def build_vocabulary(texts) -> Counter:
    doc_freq = Counter()
    for text in texts:
        doc_freq.update({w.lower() for w in WORD_RE.findall(normalize_text(text))})
    return doc_freq


def main():
    config = load_config()
    products = pd.read_csv(project_path(config["outputs"]["products"]), dtype={"product_id": str})
    doc_freq = build_vocabulary(products["ingredients_text_raw"])
    kept = sorted((w, n) for w, n in doc_freq.items() if n >= MIN_DOC_FREQ)
    lines = [f"# words occurring in >= {MIN_DOC_FREQ} products of data/processed/products.csv",
             "# format: word<TAB>number_of_products"] + [f"{w}\t{n}" for w, n in kept]
    VOCAB_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(doc_freq):,} distinct words; kept {len(kept):,} with document frequency >= {MIN_DOC_FREQ}")


if __name__ == "__main__":
    main()
