"""Hybrid NER used by the app: dictionary rules + fine-tuned DistilBERT + fuzzy knowledge-base matching.

    from src.ner.hybrid import HybridNER
    HybridNER().predict("Sugar, palm oii, emulsifier (INS 322), suger syrup")["entities"]

Why combine them? Each component is good at something different:

  * DICTIONARY / REGEX RULES (Person 1): almost never wrong when a known term matches exactly
    ("glucose syrup", "INS 330", "refined palm oil" via the head-word rule). They also know which
    text is NOT an ingredient (allergen advice, "PHENYLKETONURICS: contains phenylalanine").
  * DISTILBERT (Person 2): learned from 15,000 silver lists, so it generalises to spellings and
    OCR errors no dictionary contains ("sugor", "palrn oil", a new sweetener name).
  * FUZZY MATCHING against the knowledge base (Person 3): catches what both missed when a piece of
    text is only 1-2 letters away from a known sugar/fat/additive.

Decision rules (applied per entity, in order):
  1. A dictionary entity from an exact term, a code regex or a head-word rule is kept as it is.
  2. A dictionary FALLBACK entity (an unknown piece, labelled INGREDIENT only because it is a list
     item) takes the label of an overlapping DistilBERT entity with a more specific category.
  3. Still INGREDIENT -> fuzzy match against the specific categories of the knowledge base
     (similarity >= 90/100); a match gives that category.
  4. A DistilBERT entity where the dictionary found nothing is added when it has a specific
     category and does not lie inside an allergen/advice statement.
  5. Context: an INGREDIENT inside the brackets of a declared class takes that class's category
     ("Colour (Caramel)" -> COLOUR, "Emulsifier: Lecithins" -> ADDITIVE), and an oil source in the
     brackets of an oil ("Edible vegetable oil (Palm)") is a FAT.
  6. A list item that nobody labelled (the rules skip pieces with unknown words, e.g. OCR slips such
     as "glucose syrop") is fuzzy-matched as in step 3.

Every entity keeps a `source` field (dictionary / model / fuzzy), so the app can show why a label was
chosen - important for explaining results in the viva.
"""
import re
from functools import lru_cache

from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

from src.labeling.lexicon import load_lexicon, lookup_key
from src.labeling.rules import FAT_HEADS, SKIP_PIECE_RE, SUGAR_HEADS
from src.labeling.weak_labeler import corpus_vocabulary, find_statement_spans, label_text

SPECIFIC = {"SUGAR", "SWEETENER", "FAT", "PRESERVATIVE", "COLOUR", "ADDITIVE", "FLAVOURING", "INS_CODE"}
FUZZY_LABELS = ("SUGAR", "SWEETENER", "FAT", "PRESERVATIVE", "COLOUR", "ADDITIVE", "FLAVOURING")
FUZZY_CUTOFF = 90
PIECE_RE = re.compile(r"[^,;:()\[\]{}]+")
# words that look like a sugar/fat head word but are not (fuzzy matching must not touch them)
FUZZY_STOP = {"salt", "malt", "soya", "sago", "suji", "milk", "silk", "oat", "oats", "water", "spices", "spice"}


@lru_cache(maxsize=1)
def fuzzy_terms() -> dict:
    """Specific-category dictionary terms (>= 5 letters) -> (label, INS number)."""
    terms = {}
    for term, entry in load_lexicon().items():
        if entry.label in FUZZY_LABELS and len(term) >= 5:
            terms[term] = (entry.label, entry.number)
    return terms


def fuzzy_label(text: str, cutoff: int = FUZZY_CUTOFF):
    """('SUGAR', 'glucose syrup', '', 0.93) for 'glucose syrop', or None."""
    key = lookup_key(text)
    if len(key) < 5 or key in FUZZY_STOP:
        return None
    terms = fuzzy_terms()
    match = process.extractOne(key, list(terms), scorer=fuzz.ratio, score_cutoff=cutoff)
    if match:
        label, number = terms[match[0]]
        return label, match[0], number, round(match[1] / 100, 3)
    # head-word rule with an OCR slip in the last word: "refined palm oii" -> FAT, "invert syrop" -> SUGAR
    words = key.split()
    # (only for a last word that is NOT itself a real ingredient word: "rolled oat" must stay oats)
    if len(words) >= 2 and len(words[-1]) >= 3 and words[-1] not in corpus_vocabulary():
        last = words[-1]
        for heads, label in ((SUGAR_HEADS, "SUGAR"), (FAT_HEADS, "FAT")):
            for head in heads:
                if abs(len(head) - len(last)) <= 1 and Levenshtein.distance(last, head) <= (1 if len(head) <= 6 else 2):
                    return label, head, "", round(fuzz.ratio(last, head) / 100, 3)
    return None


CLASS_TO_LABEL = {"colour": "COLOUR", "preservative": "PRESERVATIVE", "sweetener": "SWEETENER"}
NOT_ADDITIVE_CLASSES = {"carrier", "bulking-agent", "packaging-gas", "propellent-gas"}
OIL_SOURCES = {"palm", "palmolein", "palm olein", "sunflower", "soybean", "soyabean", "soya", "soy", "cottonseed",
               "rapeseed", "canola", "groundnut", "peanut", "coconut", "rice bran", "mustard", "olive", "corn",
               "maize", "sesame", "safflower", "palm kernel", "shea", "sal", "kokum", "mango kernel"}
OPENERS = {"(": ")", "[": "]"}


def enclosing_head(text: str, start: int):
    """Start offset of the text that a bracket (or 'Class:') around position `start` belongs to:
    'Colour (Caramel E150d)' -> the offset of 'Colour'. None when the entity is not inside one."""
    depth, same_item = 0, True
    for i in range(start - 1, -1, -1):
        ch = text[i]
        if ch in ")]":
            depth += 1
        elif ch in "([":
            if depth == 0:
                return i
            depth -= 1
        elif ch in ",;" and depth == 0:
            same_item = False
        elif ch == ":" and depth == 0 and same_item:
            # 'Emulsifier: Lecithins' - a colon in the same list item also introduces a class
            return i
    return None


# basic foods that are never an additive, whatever bracket they appear in
NEVER_ADDITIVE_RE = re.compile(r"\b(?:salt|yeast|water|flour|milk|starch|spices?|sugar|vanilla|"
                               r"nature[\s-]identical|cocoa|wheat|rice|corn|soya?|egg|butter|cream)\b", re.IGNORECASE)


def bracket_closes(text: str, opener: int) -> int:
    """Position of the bracket that closes text[opener], or -1 if it is never closed (an OCR/typing slip:
    then we cannot know where the class's list ends, so the context rule is not applied)."""
    if text[opener] not in OPENERS:
        return len(text)                     # 'Class:' - ends at the end of the list item
    depth = 0
    for i in range(opener, len(text)):
        if text[i] in "([":
            depth += 1
        elif text[i] in ")]":
            depth -= 1
            if depth == 0:
                return i
    return -1


def relabel_by_context(text: str, entities: list) -> None:
    """In-place: INGREDIENT entities inside a declared class (or an oil's brackets) get a category."""
    for k, ent in enumerate(entities):
        if ent["label"] != "INGREDIENT" or NEVER_ADDITIVE_RE.search(ent["text"]):
            continue
        opener = enclosing_head(text, ent["start"])
        if opener is None or bracket_closes(text, opener) < ent["end"]:
            continue
        before = [e for e in entities[:k] if e["end"] <= opener and not text[e["end"]:opener].strip(" ")]
        if not before:
            continue
        head = before[-1]
        if head["label"] == "FUNCTION_CLASS":
            class_id = head.get("class_id", "")
            if not class_id:
                continue
            label = CLASS_TO_LABEL.get(class_id, "ADDITIVE" if class_id not in NOT_ADDITIVE_CLASSES else None)
            if label:
                ent.update(label=label, rule=f"context:{class_id}", source="context")
        elif head["label"] == "FAT" and text[opener] in "([" and lookup_key(ent["text"]) in OIL_SOURCES:
            ent.update(label="FAT", rule="context:oil-source", source="context")


def looks_like_item(piece: str) -> bool:
    """A short, mostly alphabetic list item ("maltodextrn", "dried mango") -> treat as an ingredient."""
    if SKIP_PIECE_RE.match(piece.strip()):
        return False
    words = piece.split()
    letters = sum(ch.isalpha() for ch in piece)
    return 1 <= len(words) <= 4 and letters >= 4 and letters / max(1, len(piece.replace(" ", ""))) >= 0.8


def _overlaps(a, b) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


class HybridNER:
    def __init__(self, model=None):
        """model: a TransformerTagger (or anything with .predict(text)), or None for rules + fuzzy only."""
        self.model = model
        self.name = "hybrid (rules + " + ("DistilBERT" if model is not None else "no model") + " + fuzzy KB)"

    def predict(self, text: str) -> dict:
        rules = label_text(text)
        norm = rules["text"]
        model_entities = []
        if self.model is not None:
            predicted = self.model.predict(text)
            if predicted["text"] == norm:                      # same normalisation -> offsets comparable
                model_entities = predicted["entities"]
        statements = find_statement_spans(norm)

        final = []
        for ent in rules["entities"]:
            ent = dict(ent)
            if not ent.get("rule", "").startswith("fallback"):
                ent["source"] = "dictionary"
                final.append(ent)
                continue
            # unknown list item: ask the model, then the fuzzy matcher
            specific = [m for m in model_entities if _overlaps(m, ent) and m["label"] in SPECIFIC]
            if specific:
                for m in specific:
                    final.append({**m, "rule": "model", "source": "model"})
                continue
            hit = fuzzy_label(ent["text"])
            if hit:
                label, term, number, score = hit
                ent.update(label=label, rule=f"fuzzy:{term}", source="fuzzy", fuzzy_term=term, fuzzy_score=score)
                if number:
                    ent["number"] = number
            else:
                ent["source"] = "dictionary"
            final.append(ent)

        # model entities in places where the rules found nothing
        for m in model_entities:
            if m["label"] not in SPECIFIC or any(_overlaps(m, f) for f in final):
                continue
            if any(s <= m["start"] and m["end"] <= e for s, e in statements):
                continue
            if not re.search(r"[A-Za-z0-9]{2}", m["text"]):
                continue
            final.append({**m, "rule": "model", "source": "model"})

        # list items nobody labelled
        for m in PIECE_RE.finditer(norm):
            start, end = m.start(), m.end()
            piece = norm[start:end]
            lead = len(piece) - len(piece.lstrip(" .-&"))
            start, end = start + lead, start + len(piece.rstrip(" .-&*"))
            if end - start < 5 or any(f["start"] < end and start < f["end"] for f in final):
                continue
            if any(s <= start and end <= e for s, e in statements):
                continue
            piece = norm[start:end]
            hit = fuzzy_label(piece)
            if hit:
                label, term, number, score = hit
                final.append({"start": start, "end": end, "text": piece, "label": label,
                              "rule": f"fuzzy:{term}", "source": "fuzzy", "fuzzy_term": term,
                              "fuzzy_score": score, **({"number": number} if number else {})})
            elif looks_like_item(piece):
                final.append({"start": start, "end": end, "text": piece, "label": "INGREDIENT",
                              "rule": "list_item", "source": "list item"})

        # model entities: link through the closest dictionary term when one is close enough
        # (a close dictionary term also wins over the model's label: "dextrcse" -> dextrose -> SUGAR)
        for f in final:
            if f["source"] == "model" and f["label"] != "INS_CODE":
                hit = fuzzy_label(f["text"], cutoff=85)
                if hit:
                    f["label"], f["fuzzy_term"] = hit[0], hit[1]
                    if hit[2]:
                        f["number"] = hit[2]

        final.sort(key=lambda e: e["start"])
        relabel_by_context(norm, final)
        rules["entities"] = final
        return rules
