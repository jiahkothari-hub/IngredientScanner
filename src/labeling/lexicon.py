"""Load all dictionaries into ONE lookup table:  lower-cased term -> LexiconEntry.

Sources, from most to least trusted (a term already defined by a more trusted source is not
overwritten by a less trusted one):
  1. curated lists in configs/lexicons/*.txt   (sugars, sweeteners, fats, flavourings, function classes)
  2. configs/lexicons/additive_aliases.txt     (common label names -> INS number)
  3. Open Food Facts additive taxonomy         (data/processed/additives_reference.csv)

Named additives get their label from their INS number (configs/lexicons/additive_label_rules.yaml):
COLOUR, PRESERVATIVE, SWEETENER, otherwise ADDITIVE.
"""
import re
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
import yaml

from src.utils.config import project_path

LEXICON_DIR = project_path("configs/lexicons")
ADDITIVES_CSV = project_path("data/processed/additives_reference.csv")


@dataclass(frozen=True)
class LexiconEntry:
    term: str            # lower-cased surface form, e.g. "glucose syrup"
    label: str           # NER label, e.g. "SUGAR"
    source: str          # which dictionary it came from (used as the rule name in silver data)
    number: str = ""     # INS/E number for additives ("330"), else ""
    class_id: str = ""   # functional class id for FUNCTION_CLASS terms ("acidity-regulator")


def lookup_key(term: str) -> str:
    """Key used for dictionary lookups: lower case, single spaces, no spaces around hyphens.
    'Mono - and Diglycerides' and 'mono- and diglycerides' both become 'mono-and diglycerides'."""
    term = re.sub(r"\s+", " ", term.lower()).strip()
    term = re.sub(r"\s*-\s*", "-", term)
    # US colour names: "Yellow #6", "Red No. 40", "FD&C Red 40" -> "yellow 6", "red 40", "fd&c red 40"
    return re.sub(r"\b(red|yellow|blue|green)\s*(?:#|no\.?)\s*(\d)", r"\1 \2", term)


def read_term_file(name: str) -> list:
    """Read a lexicon .txt file: one entry per line, '#' starts a comment, '|' separates columns."""
    rows = []
    for line in (LEXICON_DIR / name).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip() if not line.lstrip().startswith("#") else ""
        if line:
            rows.append([part.strip() for part in line.split("|")])
    return rows


@lru_cache(maxsize=1)
def additive_rules() -> dict:
    with open(LEXICON_DIR / "additive_label_rules.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _in_rule(base: int, rule: dict) -> bool:
    in_range = any(lo <= base <= hi for lo, hi in rule.get("ranges", []))
    return (in_range or base in rule.get("extra", [])) and base not in rule.get("exclude", [])


def label_for_number(number: str) -> str:
    """NER label for a named additive with this INS number: '102' -> COLOUR, '211' -> PRESERVATIVE."""
    digits = re.match(r"\d+", str(number))
    if not digits:
        return "ADDITIVE"
    base = int(digits.group(0))
    rules = additive_rules()
    for label in ("colour", "preservative", "sweetener"):
        if _in_rule(base, rules[label]):
            return label.upper()
    return "ADDITIVE"


def plural_forms(term: str) -> list:
    """'emulsifier' -> ['emulsifier', 'emulsifiers'];  'raising agent' -> [..., 'raising agents']."""
    forms = [term]
    if not term.endswith("s"):
        forms.append(term + "s")
    return forms


def _usable_taxonomy_name(name: str, blocked: set) -> bool:
    lowered = name.lower()
    return (
        len(lowered) >= 4
        and lowered not in blocked
        and not re.fullmatch(r"[e\s\d\-()a-z]{0,2}\d+.*", lowered)  # codes such as "E330", "330"
        and "cas " not in lowered                                    # "CAS 8028-89-5"
        and len(lowered.split()) <= 7
    )


@lru_cache(maxsize=1)
def load_lexicon() -> dict:
    lexicon = {}

    def add(term, label, source, number="", class_id=""):
        term = lookup_key(term)
        if term and term not in lexicon:          # first (= most trusted) definition wins
            lexicon[term] = LexiconEntry(term, label, source, number, class_id)

    blocked = {b.lower() for b in additive_rules()["blocked_names"]}
    additives = pd.read_csv(ADDITIVES_CSV, dtype=str).fillna("")
    # name -> INS number from the taxonomy, so curated terms such as "sucralose" are linked to 955
    number_of_name = {}
    for row in additives.itertuples():
        for name in [row.name] + [s for s in row.synonyms.split("|") if s]:
            number_of_name.setdefault(lookup_key(name), row.number)

    # 1. curated lists
    for file_name, label in [("sweeteners.txt", "SWEETENER"), ("sugars.txt", "SUGAR"),
                             ("fats.txt", "FAT"), ("flavourings.txt", "FLAVOURING")]:
        for (term, *_) in read_term_file(file_name):
            number = number_of_name.get(lookup_key(term), "") if label == "SWEETENER" else ""
            add(term, label, f"dict:{file_name[:-4]}", number=number)
    for term, class_id in read_term_file("function_classes.txt"):
        for form in plural_forms(term):
            add(form, "FUNCTION_CLASS", "dict:function_classes", class_id=class_id)

    # 2. aliases
    for term, number in read_term_file("additive_aliases.txt"):
        add(term, label_for_number(number), "dict:additive_aliases", number=number.lower())

    # 3. Open Food Facts taxonomy names and synonyms
    for row in additives.itertuples():
        names = [row.name] + [s for s in row.synonyms.split("|") if s]
        for name in names:
            if _usable_taxonomy_name(name, blocked):
                add(name, label_for_number(row.number), "dict:off_taxonomy", number=row.number)
    return lexicon


@lru_cache(maxsize=1)
def additive_reference() -> dict:
    """number -> {'name': ..., 'function_classes': [...]} for linking codes to names."""
    additives = pd.read_csv(ADDITIVES_CSV, dtype=str).fillna("")
    return {
        row.number: {"name": row.name, "function_classes": [c for c in row.function_classes.split("|") if c]}
        for row in additives.itertuples()
    }


def function_class_terms() -> list:
    """All FUNCTION_CLASS surface forms (used by the additive-code detector for context)."""
    return sorted(t for t, e in load_lexicon().items() if e.label == "FUNCTION_CLASS")
