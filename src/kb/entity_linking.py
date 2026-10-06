"""Entity linking: connect each detected entity to an entry of the knowledge base.

    from src.kb.entity_linking import EntityLinker
    EntityLinker().link({"text": "INS 330", "label": "INS_CODE"})
    -> {"kb_id": "INS:330", "canonical_name": "Citric acid", "function": "Antioxidant, Sequestrant", ...,
        "link_method": "ins_number", "link_score": 1.0}

Linking methods, tried in this order:
  1. ins_number  - the entity is a code ("INS 330", "E 150d", "(471)") -> look the number up
                   (sub-types fall back to their parent: 500ii -> 500)
  2. exact_name  - the normalised text is a known term ("soy lecithin" -> INS 322)
  2b. additive_name - an additive entity named like an official additive name or synonym in the KB
                   ("riboflavin" used as a colour -> INS 101)
  3. fuzzy_name  - close spelling of a known term, for OCR errors ("lecithln" -> "lecithin"),
                   Levenshtein similarity >= 88/100, searched within the entity's own category
  4. category    - nothing specific found: explain the category only ("A food ingredient ...")
"""
from functools import lru_cache

import pandas as pd
from rapidfuzz import fuzz, process

from src.labeling.lexicon import lookup_key
from src.preprocessing.additive_codes import normalize_code
from src.utils.config import project_path

KB_DIR = project_path("data/knowledge_base")
FUZZY_CUTOFF = 88
ADDITIVE_LABELS = {"ADDITIVE", "PRESERVATIVE", "COLOUR", "SWEETENER", "INS_CODE"}


class EntityLinker:
    def __init__(self, kb_dir=KB_DIR):
        self.additives = pd.read_csv(kb_dir / "additives.csv", dtype=str).fillna("").set_index("ins_number")
        terms = pd.read_csv(kb_dir / "ingredient_terms.csv", dtype=str).fillna("")
        self.terms = {row.term: row for row in terms.itertuples()}
        self.terms_by_category = terms.groupby("category")["term"].apply(list).to_dict()
        classes = pd.read_csv(kb_dir / "function_classes.csv", dtype=str).fillna("")
        self.classes = classes.set_index("class_id")
        # official additive names and synonyms -> INS number ("riboflavin" -> 101), for additive entities
        self.additive_names = {}
        for number, row in self.additives.iterrows():
            for name in [row["name"], *row["synonyms"].split("|")]:
                key = lookup_key(name) if name else ""
                if key and key not in self.additive_names:
                    self.additive_names[key] = number
        cats = pd.read_csv(kb_dir / "categories.csv", dtype=str)
        self.category_text = dict(zip(cats["category"], cats["description"]))

    # -------------------------------------------------------------------- helpers
    def _additive(self, number: str):
        for candidate in (number, number.rstrip("ivx"), "".join(ch for ch in number if ch.isdigit())):
            if candidate in self.additives.index:
                return candidate, self.additives.loc[candidate]
        return None, None

    def _from_additive(self, number, row, method, score):
        return {"kb_id": row["kb_id"], "canonical_name": row["name"], "ins_number": number,
                "kb_category": row["ner_label"], "function": row["function_classes_text"],
                "description": row["function_description"] or self.category_text.get(row["ner_label"], ""),
                "more_info_url": row["wikidata_url"], "link_method": method, "link_score": score}

    def _from_term(self, term_row, label, method, score):
        if term_row.ins_number:
            number, row = self._additive(term_row.ins_number)
            if row is not None:
                return self._from_additive(number, row, method, score)
        if term_row.class_id and term_row.class_id in self.classes.index:
            cls = self.classes.loc[term_row.class_id]
            return {"kb_id": f"CLASS:{term_row.class_id}", "canonical_name": cls["name"], "ins_number": "",
                    "kb_category": "FUNCTION_CLASS", "function": cls["name"],
                    "description": cls["description"] or self.category_text["FUNCTION_CLASS"],
                    "more_info_url": "", "link_method": method, "link_score": score}
        return {"kb_id": f"TERM:{term_row.term}", "canonical_name": term_row.term, "ins_number": "",
                "kb_category": term_row.category, "function": "",
                "description": self.category_text.get(term_row.category, ""), "more_info_url": "",
                "link_method": method, "link_score": score}

    # -------------------------------------------------------------------- main API
    def link(self, entity: dict) -> dict:
        text, label = entity["text"], entity["label"]
        number = entity.get("number") or (normalize_code(text) if label == "INS_CODE" else None)
        if number:
            found, row = self._additive(number)
            if row is not None:
                return self._from_additive(found, row, "ins_number", 1.0)

        key = lookup_key(text)
        if key in self.terms:
            return self._from_term(self.terms[key], label, "exact_name", 1.0)

        if label in ADDITIVE_LABELS and key in self.additive_names:
            found, row = self._additive(self.additive_names[key])
            if row is not None:
                return self._from_additive(found, row, "additive_name", 1.0)

        candidates = self.terms_by_category.get(label, []) if label != "INGREDIENT" else []
        if label in ADDITIVE_LABELS:
            candidates = [t for c in ("ADDITIVE", "PRESERVATIVE", "COLOUR", "SWEETENER") for t in self.terms_by_category.get(c, [])]
        if candidates and len(key) >= 4:
            match = process.extractOne(key, candidates, scorer=fuzz.ratio, score_cutoff=FUZZY_CUTOFF)
            if match:
                return self._from_term(self.terms[match[0]], label, "fuzzy_name", round(match[1] / 100, 3))

        return {"kb_id": "", "canonical_name": "", "ins_number": "", "kb_category": label, "function": "",
                "description": self.category_text.get(label, ""), "more_info_url": "",
                "link_method": "category", "link_score": 0.0}


@lru_cache(maxsize=1)
def get_linker() -> EntityLinker:
    return EntityLinker()
