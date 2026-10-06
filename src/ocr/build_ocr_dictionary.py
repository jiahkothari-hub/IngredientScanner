"""Build the food-domain spelling dictionary used to correct OCR text.

Usage:  python -m src.ocr.build_ocr_dictionary

Words and their frequencies come from the 22,000 ingredient lists in data/processed/products.csv
(words seen in at least 3 lists), plus every word of the project's lexicon (sugars, fats, additives,
functional classes ...), which get a high count so they are preferred. Typos that are frequent enough to
pass the threshold ("vegetabie", "buter") are removed: a rare word is dropped when a word one letter away
is at least 20 times more frequent. Output: data/processed/ocr_dictionary.txt ("word<TAB>count" per line).
"""
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from src.labeling.lexicon import load_lexicon
from src.utils.config import load_config, project_path

OUT = project_path("data/processed/ocr_dictionary.txt")
EXTRA_WORDS = """ingredients ingredient contains may traces allergen allergens nutrition energy protein carbohydrate
carbohydrates sugars fat fats saturated fibre fiber sodium salt per serving serve servings package kcal
phenylketonurics phenylalanine permitted synthetic natural identical nature artificial flavouring flavourings
flavoring flavorings emulsifier emulsifiers stabiliser stabilizer stabilisers stabilizers thickener thickeners
regulator regulators acidity raising leavening agents agent anticaking antioxidant antioxidants preservative
preservatives colour colours color colors sweetener sweeteners humectant glazing gelling improver conditioner
added spices condiments seasoning masala maida atta besan suji rava jaggery ghee vanaspati palmolein""".split()


def find_typos(words: dict, protected: set) -> set:
    """Rare words with a 20x more frequent neighbour at edit distance 1 (vegetabie -> vegetable)."""
    from symspellpy import SymSpell, Verbosity
    sym = SymSpell(max_dictionary_edit_distance=1, prefix_length=7)
    for w, c in words.items():
        sym.create_dictionary_entry(w, c)
    typos = set()
    for w, c in words.items():
        if w in protected or c >= 200 or len(w) < 4:
            continue
        for s in sym.lookup(w, Verbosity.ALL, max_edit_distance=1):
            if s.term != w and s.count >= 20 * c:
                typos.add(w)
                break
    return typos


def main():
    config = load_config()
    texts = pd.read_csv(project_path(config["outputs"]["products"]), usecols=["ingredients_text_raw"])
    counts = Counter(w for t in texts["ingredients_text_raw"] for w in re.findall(r"[a-z]+", str(t).lower()))
    words = {w: c for w, c in counts.items() if c >= 3 and len(w) >= 2}
    boost = max(words.values())
    for term in load_lexicon():
        for w in re.findall(r"[a-z]+", term):
            if len(w) >= 2:
                words[w] = max(words.get(w, 0), boost // 10)
    for w in EXTRA_WORDS:
        words[w] = max(words.get(w, 0), boost // 10)
    import symspellpy                       # real English words are never treated as typos
    english = Path(symspellpy.__file__).parent / "frequency_dictionary_en_82_765.txt"
    protected = {w for term in load_lexicon() for w in re.findall(r"[a-z]+", term)} | set(EXTRA_WORDS)
    protected |= {line.split(" ", 1)[0] for line in english.read_text(encoding="utf-8").splitlines()}
    typos = find_typos(words, protected)
    for w in typos:
        del words[w]
    print(f"Removed {len(typos):,} likely typos, e.g. {sorted(typos)[:12]}")
    OUT.write_text("".join(f"{w}\t{c}\n" for w, c in sorted(words.items())), encoding="utf-8")
    print(f"Saved {len(words):,} words -> {OUT.relative_to(project_path(''))}")


if __name__ == "__main__":
    main()
