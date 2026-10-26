"""What the assistant knows, and how it finds the ingredient a question is about.

    from src.assistant.knowledge import KnowledgeBase
    kb = KnowledgeBase()
    kb.find_mentions("is E471 vegan?")     -> [Mention(kind="additive", key="471", name="Mono- and diglycerides...")]
    kb.retrieve("the stuff that makes it thick")  -> best matching knowledge-base entries (TF-IDF)

Sources, all already part of the project (data/knowledge_base/):
  additives.csv         731 additives: INS/E number, name, synonyms, function, description, vegan/vegetarian
  ingredient_terms.csv  1,563 ingredient names -> category (SUGAR, FAT, ...) and linked additive
  function_classes.csv  what "emulsifier", "acidity regulator", ... mean
  categories.csv        what a sugar / sweetener / fat ... is
  additive_adi.csv      acceptable daily intakes (via src.app.intake.additive_guidance)

Finding the mention (information extraction on the question), in order:
  1. additive codes ("INS 330", "E150d", "471")          src.preprocessing.additive_codes
  2. exact names, longest match first ("soy lecithin" before "lecithin"), including the words of the
     scanned label, so "the dextrose in this" points at the product's own entity
  3. fuzzy names for typos ("malodextrin"), Levenshtein ratio >= 88
Retrieval (when no name is found): TF-IDF over every entry's name, synonyms and description, cosine
similarity, so a description of an ingredient can still find it.
"""
import re
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd
from rapidfuzz import fuzz, process

from src.labeling.lexicon import lookup_key, read_term_file
from src.preprocessing.additive_codes import normalize_code
from src.utils.config import project_path

KB_DIR = project_path("data/knowledge_base")
LEXICONS = {"sugars.txt": "SUGAR", "sweeteners.txt": "SWEETENER", "fats.txt": "FAT", "flavourings.txt": "FLAVOURING"}
# the scanner's "sugar-like carbohydrates" (src/app/scanner.py), explained with the app's own wording
SUGAR_LIKE = ["maltodextrin", "dextrin", "glucose solids", "corn syrup solids", "corn solids", "rice solids"]
SUGAR_LIKE_TEXT = ("A sugar-like carbohydrate made from starch. It is not counted as 'sugars' on the nutrition label, "
                   "but it is digested quickly into glucose.")
# additive_regulatory.csv: additives whose rules differ between the EU, India (FSSAI) and the US (FDA)
REGIONS = {"EU": "eu", "India": "india", "US": "us"}
STATUS_FLAG = {"banned": "Banned in {region}", "not_authorised": "Not authorised in {region}",
               "phasing_out": "Being phased out in {region}", "warning_label": "Warning label required in {region}",
               "restricted": "Restricted use in {region}"}
STATUS_TEXT = {"permitted": "permitted", "banned": "banned", "not_authorised": "not authorised",
               "phasing_out": "being phased out", "warning_label": "permitted with a warning label",
               "restricted": "permitted for limited uses", "not_checked": "not checked"}
CODE_RE = re.compile(r"\b(?:ins|e)\s*-?\s*\d{3,4}\s*[a-z]?(?:\s*\(?(?:iii|ii|iv|vi|v|i)\)?)?(?![\w%])|"
                     r"(?<![\w.,])\d{3,4}[a-z]?(?!\s*(?:g|mg|ml|kg|kcal|kj|%|\.\d))\b", re.IGNORECASE)
FUZZY_CUTOFF = 88
STOPWORDS = set("""a an the is are was it this that these those in on of for to do does did what which why how
    there any have has with my me i you your can could should would will be please tell about explain product
    food contain contains containing used use added add here its it's whats what's thing stuff""".split())

# words in a question that name a whole category of the product's ingredients
CATEGORY_WORDS = [
    ("hidden_sugar", r"hidden\s+sugars?|sugars?\s+(?:under|by)\s+other\s+names|other\s+names\s+for\s+sugar"),
    # sub-groups of the generic ingredients (src/kb/ingredient_subgroups.py): "SUB:<key>"
    ("SUB:culture", r"probiotic|\bcultures?\b|bacteria|microbe|live\s+yeast"),
    ("SUB:enzyme", r"enzymes?"),
    ("SUB:stimulant", r"stimulant|caffeine"),
    ("SUB:protein", r"proteins?(?:\s+sources?)?\b"),
    ("SUB:fibre", r"fib(?:re|er)s?\b"),
    ("SUB:vitamin_mineral", r"vitamins?|minerals?|fortif"),
    ("SUB:cocoa_coffee", r"cocoa|coffee|chocolate|\btea\b"),
    ("SUB:legume", r"legumes?|pulses?|\bbeans?\b|lentils?"),
    ("SUB:nut_seed", r"nuts?\s+(?:and|&)\s+seeds?|\bseeds?\b"),
    ("SUB:grain", r"grains?|cereals?|flours?|starch(?:es)?"),
    ("SUB:spice_herb", r"spices?|herbs?|masala"),
    ("SUB:fruit_veg", r"fruits?|vegetables?|\bveg(?:gies)?\b"),
    ("SUB:dairy", r"dairy\s+ingredients?|milk\s+ingredients?|milk\s+products?"),
    ("SUB:seasoning", r"seasonings?|yeast\s+extract|\bextracts?\b"),
    ("SWEETENER", r"sweeteners?|artificial\s+sweet"),
    ("SUGAR", r"\bsugars?\b|syrups?"),
    ("FAT", r"\bfats?\b|\boils?\b"),
    ("PRESERVATIVE", r"preservatives?"),
    ("COLOUR", r"colou?rs?|colou?rings?|dyes?"),
    ("FLAVOURING", r"flavou?rs?|flavou?rings?"),
    ("CODE", r"e[\s-]?numbers?|codes?|ins\s+numbers?"),
    ("ADDITIVE", r"additives?|artificial|chemicals?"),
    ("ALL", r"all\s+(?:the\s+)?ingredients|every\s+ingredient|full\s+list"),
]
NUTRIENT_WORDS = [("saturated_fat_g", r"saturat"), ("sugars_g", r"sugar|sweet|teaspoon"), ("salt_g", r"salt|sodium"),
                  ("fat_g", r"\bfat"), ("energy_kcal", r"calor|kcal|energy"), ("carbohydrate_g", r"carb"),
                  ("protein_g", r"protein"), ("fibre_g", r"fib(?:re|er)")]
# not in the additive table, but common animal-derived ingredients (for "is it vegan?")
ANIMAL = re.compile(r"\b(?:milk|whey|casein|caseinate|lactose|butter|ghee|cream|cheese|yog(?:h)?urt|curd|paneer|egg|"
                    r"albumen|honey|gelatin[e]?|beef|pork|chicken|meat|fish|anchov|lard|tallow|shellac|carmine|"
                    r"cochineal|collagen|isinglass)\w*", re.IGNORECASE)
NOT_VEGETARIAN = re.compile(r"\b(?:gelatin[e]?|beef|pork|chicken|meat|fish|anchov\w*|lard|tallow|carmine|cochineal|"
                            r"collagen|isinglass)\b", re.IGNORECASE)


@dataclass
class Mention:
    kind: str                     # additive | term | class | entity
    key: str                      # INS number, term, class id or entity index
    name: str                     # display name
    span: str = ""                # the words in the question
    method: str = "exact"         # code | exact | fuzzy | retrieval | reference
    extra: dict = field(default_factory=dict)


class KnowledgeBase:
    def __init__(self, kb_dir=KB_DIR):
        self.additives = pd.read_csv(kb_dir / "additives.csv", dtype=str).fillna("").set_index("ins_number")
        self.terms = pd.read_csv(kb_dir / "ingredient_terms.csv", dtype=str).fillna("").set_index("term")
        self.classes = pd.read_csv(kb_dir / "function_classes.csv", dtype=str).fillna("").set_index("class_id")
        self.categories = pd.read_csv(kb_dir / "categories.csv", dtype=str).fillna("").set_index("category")
        self.names = {}               # lookup key -> (kind, key, display name)
        for number, row in self.additives.iterrows():
            for n in [row["name"]] + [s for s in row["synonyms"].split("|") if s]:
                self.names.setdefault(lookup_key(n), ("additive", number, row["name"]))
        for term, row in self.terms.iterrows():
            if row["ins_number"] and row["ins_number"] in self.additives.index:
                self.names.setdefault(term, ("additive", row["ins_number"], self.additives.loc[row["ins_number"], "name"]))
            elif row["class_id"] and row["class_id"] in self.classes.index:
                self.names.setdefault(term, ("class", row["class_id"], self.classes.loc[row["class_id"], "name"]))
            else:
                self.names.setdefault(term, ("term", term, term))
        for class_id, row in self.classes.iterrows():
            for n in [row["name"]] + [s for s in row["synonyms"].split("|") if s]:
                self.names.setdefault(lookup_key(n), ("class", class_id, row["name"]))
        self.lexicon_category = {}    # the NER lexicons: sugars, sweeteners, fats, flavourings
        for file, category in LEXICONS.items():
            for row in read_term_file(file):
                term = lookup_key(row[0] if isinstance(row, (list, tuple)) else row)
                self.lexicon_category.setdefault(term, category)
                self.names.setdefault(term, ("term", term, term))
        for term in SUGAR_LIKE:
            self.lexicon_category[term] = "SUGAR_LIKE"
            self.names[term] = ("term", term, term)
        self.regulatory = {}          # INS number -> rules in the EU, India and the US
        path = kb_dir / "additive_regulatory.csv"
        if path.exists():
            for row in pd.read_csv(path, dtype=str).fillna("").to_dict("records"):
                for number in row["ins_numbers"].split("|"):
                    self.regulatory[number] = row
                    if number in self.additives.index:
                        for alias in [a for a in row["aliases"].split("|") if a]:
                            self.names.setdefault(lookup_key(alias), ("additive", number, self.additives.loc[number, "name"]))
        self.max_words = max(len(k.split()) for k in self.names)

    # ---------------------------------------------------------------------------------- mentions
    def find_mentions(self, question: str, entities: list = ()) -> list:
        """Ingredients named in the question, in order of appearance (codes, exact, then fuzzy names)."""
        found, taken = [], []
        for m in CODE_RE.finditer(question):
            number = normalize_code(m.group(0))
            base = re.match(r"\d+[a-z]?", number or "")
            for candidate in (number, base and base.group(0), base and re.match(r"\d+", base.group(0)).group(0)):
                if candidate and candidate in self.additives.index:
                    found.append((m.start(), Mention("additive", candidate, self.additives.loc[candidate, "name"],
                                                     m.group(0).strip(), "code")))
                    taken.append((m.start(), m.end()))
                    break
        words = [(w.group(0), w.start(), w.end()) for w in re.finditer(r"[\w'&-]+", question.lower())]
        entity_keys = {lookup_key(e["text"]): i for i, e in enumerate(entities)}
        i = 0
        while i < len(words):
            match = None
            for n in range(min(self.max_words, len(words) - i), 0, -1):
                start, end = words[i][1], words[i + n - 1][2]
                if any(s < end and start < e for s, e in taken):
                    continue
                key = lookup_key(" ".join(w for w, _, _ in words[i:i + n]))
                if n == 1 and (key in STOPWORDS or len(key) < 3):
                    continue
                if key in entity_keys and entities[entity_keys[key]].get("label") not in ("FUNCTION_CLASS",):
                    ent = entities[entity_keys[key]]
                    match = (n, self._entity_mention(ent, entity_keys[key], key))
                elif key in self.names:
                    kind, k, name = self.names[key]
                    match = (n, Mention(kind, k, name, key, "exact"))
                if match:
                    break
            if match:
                n, mention = match
                found.append((words[i][1], mention))
                taken.append((words[i][1], words[i + n - 1][2]))
                i += n
            else:
                i += 1
        if not found:
            found = self._fuzzy(words)
        found.sort(key=lambda x: x[0])
        unique, seen = [], set()
        for _, m in found:
            if (m.kind, m.key) not in seen:
                seen.add((m.kind, m.key))
                unique.append(m)
        return unique

    def _entity_mention(self, ent: dict, index: int, span: str) -> Mention:
        if ent.get("ins_number") and ent["ins_number"] in self.additives.index:
            return Mention("additive", ent["ins_number"], self.additives.loc[ent["ins_number"], "name"], span, "exact")
        if span in self.names:        # the knowledge base knows more than the label's generic entry
            kind, key, name = self.names[span]
            return Mention(kind, key, name, span, "exact")
        return Mention("entity", str(index), ent.get("canonical_name") or ent["text"], span, "exact",
                       {"label": ent["label"]})

    def _fuzzy(self, words: list) -> list:
        keys = list(self.names)
        out = []
        for n in (3, 2, 1):
            for i in range(len(words) - n + 1):
                phrase = " ".join(w for w, _, _ in words[i:i + n])
                if len(phrase) < 5 or all(w in STOPWORDS for w, _, _ in words[i:i + n]):
                    continue
                hit = process.extractOne(phrase, keys, scorer=fuzz.ratio, score_cutoff=FUZZY_CUTOFF)
                if hit:
                    kind, k, name = self.names[hit[0]]
                    out.append((words[i][1], Mention(kind, k, name, phrase, "fuzzy", {"score": hit[1]})))
            if out:
                return out
        return out

    # ---------------------------------------------------------------------------------- retrieval
    @lru_cache(maxsize=1)
    def _index(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        docs, refs = [], []
        for number, row in self.additives.iterrows():
            docs.append(f"{row['name']} {row['synonyms'].replace('|', ' ')} {row['function_classes_text']} "
                        f"{row['function_description']}")
            refs.append(("additive", number, row["name"]))
        for class_id, row in self.classes.iterrows():
            docs.append(f"{row['name']} {row['synonyms'].replace('|', ' ')} {row['description']}")
            refs.append(("class", class_id, row["name"]))
        # character n-grams inside words: "thick" still matches "thickener", "sweet" matches "sweetener"
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        return vectorizer, vectorizer.fit_transform(docs), refs

    def retrieve(self, question: str, k: int = 3, min_score: float = 0.2) -> list:
        vectorizer, matrix, refs = self._index()
        query = " ".join(w for w in re.findall(r"[a-z]+", question.lower()) if w not in STOPWORDS)
        if not query:
            return []
        scores = (matrix @ vectorizer.transform([query]).T).toarray().ravel()
        best = scores.argsort()[::-1][:k]
        return [Mention(refs[i][0], refs[i][1], refs[i][2], query, "retrieval", {"score": round(float(scores[i]), 3)})
                for i in best if scores[i] >= min_score]

    # ---------------------------------------------------------------------------------- facts
    def additive(self, number: str) -> dict:
        row = self.additives.loc[number]
        return {"number": number, "e_code": row["e_code"], "name": row["name"], "label": row["ner_label"],
                "functions": row["function_classes_text"], "description": row["function_description"],
                "vegan": row["vegan"], "vegetarian": row["vegetarian"], "synonyms": row["synonyms"]}

    def function_class(self, class_id: str) -> dict:
        row = self.classes.loc[class_id]
        return {"name": row["name"], "description": row["description"]}

    def term_category(self, term: str) -> str:
        if term in self.lexicon_category:
            return self.lexicon_category[term]
        return self.terms.loc[term, "category"] if term in self.terms.index else ""

    def category_text(self, category: str) -> str:
        if category == "SUGAR_LIKE":
            return SUGAR_LIKE_TEXT
        return self.categories.loc[category, "description"] if category in self.categories.index else ""

    def regulatory_status(self, number: str) -> dict:
        """{region: (status, note)} for an additive whose rules differ between regions, else {}.
        Sub-types fall back to the parent number (150d -> 150d, 503ii -> 503)."""
        row = self.regulatory.get(number) or self.regulatory.get(re.match(r"\d+[a-z]?", number or "x") and
                                                                   re.match(r"\d+[a-z]?", number).group(0) or "")
        if not row:
            return {}
        return {"sources": row["sources"].split("|"), "checked": row["checked"], "other": row["other_note"],
                **{region: (row[col], row[f"{col}_note"]) for region, col in REGIONS.items()}}

    def regulatory_flags(self, number: str) -> list:
        """Short neutral flags for the statuses worth showing ("Banned in EU"), permitted ones left out."""
        status = self.regulatory_status(number)
        return [STATUS_FLAG[status[r][0]].format(region=r) for r in REGIONS if status and status[r][0] in STATUS_FLAG]

    def class_for_function(self, function_text: str) -> list:
        """'Acidity regulator, Antioxidant' -> the descriptions of those function classes."""
        out = []
        for part in [p.strip() for p in re.split(r",|/", function_text) if p.strip()]:
            key = self.names.get(lookup_key(part))
            if key and key[0] == "class":
                out.append(self.function_class(key[1]))
        return out


def category_from_question(question: str) -> str:
    for category, pattern in CATEGORY_WORDS:
        if re.search(pattern, question, re.IGNORECASE):
            return category
    return ""


def nutrient_from_question(question: str) -> str:
    for key, pattern in NUTRIENT_WORDS:
        if re.search(pattern, question, re.IGNORECASE):
            return key
    return ""


@lru_cache(maxsize=1)
def knowledge_base() -> KnowledgeBase:
    return KnowledgeBase()
