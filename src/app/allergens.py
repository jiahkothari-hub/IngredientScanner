"""Allergen detection: which of the 14 major allergen groups does the label mention?

    from src.app.allergens import find_allergens
    find_allergens("Wheat flour, sugar, milk solids. May contain traces of nuts.")
    -> {"contains": {"Cereals containing gluten": ["wheat flour"], "Milk": ["milk solids"]},
        "may_contain": {"Tree nuts": ["nuts"]}}

The 14 groups are the ones EU Regulation 1169/2011 (Annex II) requires to be emphasised; FSSAI's
labelling regulations (2020) list the same main groups. Detection is keyword based on the full text
(ingredient list + "Contains:" / "May contain" statements). "May contain" (precautionary) mentions are
reported separately because they are not ingredients.
"""
import re

ALLERGEN_WORDS = {
    "Cereals containing gluten": r"wheat|maida|atta|suji|sooji|semolina|rava|barley|rye|oats?|spelt|kamut|gluten|"
                                 r"malt(?:ed)?|couscous|bulgur|durum",
    "Crustaceans": r"crustaceans?|shrimps?|prawns?|crabs?|lobsters?|crayfish",
    "Eggs": r"eggs?|egg\s+(?:white|yolk|powder)|albumen",
    "Fish": r"fish|anchov(?:y|ies)|tuna|salmon|cod|sardines?",
    "Peanuts": r"peanuts?|groundnuts?|arachis",
    "Soybeans": r"soy|soya|soybeans?|soyabeans?|edamame|tofu",
    "Milk": r"milk|dairy|butter|ghee|cream|cheese|whey|casein(?:ates?)?|lactose|yog(?:h)?urt|curd|paneer|khoa|khoya",
    "Tree nuts": r"nuts?|tree\s+nuts|almonds?|hazelnuts?|walnuts?|cashews?|pecans?|pistachios?|macadamias?|"
                 r"brazil\s+nuts?|badam|kaju",
    "Celery": r"celery|celeriac",
    "Mustard": r"mustard|rai",
    "Sesame": r"sesame|til|gingelly|tahini",
    "Sulphites": r"sulph?ites?|sulfites?|sulphur\s+dioxide|sulfur\s+dioxide|metabisulph?ite|metabisulfite|"
                 r"\b22[0-8]\b",
    "Lupin": r"lupin(?:e|s)?",
    "Molluscs": r"molluscs?|mussels?|oysters?|squid|clams?|octopus",
}
# words that contain an allergen word but are not that allergen
EXCEPTIONS = re.compile(r"coconut|nutmeg|butternut|cocoa\s+butter|shea\s+butter|peanut\s+free|nut\s+free|"
                        r"buckwheat|milk\s+thistle|cream\s+of\s+tartar|doughnut", re.IGNORECASE)
MAY_CONTAIN_RE = re.compile(r"(?:may\s+(?:also\s+)?contain|traces?\s+of|produced\s+in\s+a\s+facility|"
                            r"manufactured\s+in\s+a\s+facility|made\s+on\s+(?:shared\s+)?equipment)[^.]*",
                            re.IGNORECASE)


def _scan(text: str) -> dict:
    found = {}
    clean = EXCEPTIONS.sub(" ", text)
    for group, pattern in ALLERGEN_WORDS.items():
        hits = re.findall(rf"\b(?:{pattern})\b", clean, re.IGNORECASE)
        if hits:
            found[group] = sorted({h.lower() for h in hits})
    return found


STATEMENT_RE = re.compile(r"[^.\n]*\b(?:contains?|allergens?|allergy|may\s+contain|traces?)\b[^.\n]*", re.IGNORECASE)


def allergen_text(ingredient_list: str, full_text: str) -> str:
    """The text allergens are searched in: the ingredient list plus the label's allergen statements
    ("Contains: milk", "May contain nuts") - not serving suggestions or nutrition tables
    ("serve with 200 ml milk" is not an ingredient)."""
    statements = [m.group(0) for m in STATEMENT_RE.finditer(full_text or "")]
    return ingredient_list + ". " + ". ".join(statements)


def find_allergens(text: str) -> dict:
    may_parts = [m.group(0) for m in MAY_CONTAIN_RE.finditer(text)]
    main_text = MAY_CONTAIN_RE.sub(" ", text)
    contains = _scan(main_text)
    may = {g: w for g, w in _scan(" ".join(may_parts)).items() if g not in contains}
    return {"contains": contains, "may_contain": may}
