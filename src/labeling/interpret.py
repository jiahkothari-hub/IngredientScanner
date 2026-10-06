"""Interpretation layer: connect identified entities to understandable information.

    IDENTIFICATION  (NER)            "INS 330" is an INS_CODE
    INTERPRETATION  (this module)    INS 330 = Citric acid; declared on the label as "acidity regulator"
    HEALTH CLAIM    (never)          -- the project does not say whether anything is good or bad

Two kinds of function information are kept apart on purpose:
  * declared_class   - what THIS label says the additive does ("Acidity regulator (INS 330)").
                       This is the most reliable source for this product.
  * reference_classes - the possible functions listed in the Open Food Facts taxonomy for that number
                       (may be several, and the list is incomplete for some additives).
Person 3's knowledge base / entity linking can extend this; the function is deliberately simple.
"""
import re

from src.labeling.lexicon import additive_reference

ADDITIVE_TYPES = {"INS_CODE", "ADDITIVE", "PRESERVATIVE", "COLOUR", "SWEETENER"}
# text allowed between a class name and its additives: brackets, separators, "and", "&"
_BRIDGE_RE = re.compile(r"^[\s()\[\]{},:;&]*(?:(?:and|or)\b[\s()\[\]{},:;&]*)*$", re.IGNORECASE)
MAX_DISTANCE = 120


def lookup_number(number: str) -> dict:
    """'330' -> {'name': 'Citric acid', 'function_classes': [...]}. Falls back to the parent number
    for sub-types ('500ii' -> '500') and returns {} if unknown."""
    reference = additive_reference()
    if not number:
        return {}
    for candidate in (number, re.match(r"\d+[a-f]?", number).group(0), re.match(r"\d+", number).group(0)):
        if candidate in reference:
            return reference[candidate]
    return {}


def declared_classes(text: str, entities: list) -> dict:
    """Map entity index -> class_id declared next to it on the label.

    "Emulsifiers (471, soy lecithin)"  -> 471 and soy lecithin get 'emulsifier'
    "potassium sorbate (preservative)" -> potassium sorbate gets 'preservative'
    """
    result = {}
    for i, ent in enumerate(entities):
        if ent["label"] != "FUNCTION_CLASS":
            continue
        # forwards: additives inside the bracket (or after the colon) that follows the class name
        j, last_end, depth, mode = i + 1, ent["end"], 0, None
        while j < len(entities) and entities[j]["label"] in ADDITIVE_TYPES:
            between = text[last_end:entities[j]["start"]]
            if not _BRIDGE_RE.match(between) or entities[j]["start"] - ent["end"] > MAX_DISTANCE:
                break
            depth += sum(between.count(c) for c in "([{") - sum(between.count(c) for c in ")]}")
            if mode is None:                       # first bridge decides: "Class (..." or "Class: ..."
                mode = "bracket" if depth > 0 else "colon" if ":" in between else None
                if mode is None:
                    break
            elif mode == "bracket" and depth <= 0:  # the bracket after the class name was closed
                break
            result.setdefault(j, ent.get("class_id", ""))
            last_end = entities[j]["end"]
            j += 1
        # backwards: "potassium benzoate and potassium sorbate (preservatives)"
        j, first_start = i - 1, ent["start"]
        while j >= 0 and entities[j]["label"] in ADDITIVE_TYPES and "(" in text[entities[j]["end"]:ent["start"]]:
            if not _BRIDGE_RE.match(text[entities[j]["end"]:first_start]):
                break
            result.setdefault(j, ent.get("class_id", ""))
            first_start = entities[j]["start"]
            j -= 1
    return result


def interpret_entities(text: str, entities: list) -> list:
    """Return copies of the entities with interpretation fields added (only where applicable)."""
    declared = declared_classes(text, entities)
    out = []
    for i, ent in enumerate(entities):
        ent = dict(ent)
        if ent["label"] in ADDITIVE_TYPES and ent.get("number"):
            info = lookup_number(ent["number"])
            ent["linked_name"] = info.get("name", "")
            ent["reference_classes"] = info.get("function_classes", [])
        if i in declared:
            ent["declared_class"] = declared[i]
        out.append(ent)
    return out


def describe(ent: dict) -> str:
    """Plain-language interpretation, e.g. 'INS 330 -> Citric acid -> acidity regulator'.
    Describes what the ingredient IS and what it is used for; never whether it is healthy."""
    parts = [ent["text"]]
    if ent.get("linked_name") and ent["linked_name"].lower() != ent["text"].lower():
        parts.append(ent["linked_name"])
    function = ent.get("declared_class") or ", ".join(ent.get("reference_classes", [])[:2])
    if function:
        parts.append(function.replace("-", " "))
    elif ent["label"] in ("SUGAR", "SWEETENER", "FAT", "FLAVOURING", "COLOUR", "PRESERVATIVE"):
        parts.append(ent["label"].lower())
    return " -> ".join(parts)
