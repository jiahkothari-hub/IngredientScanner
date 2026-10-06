"""Regex and head-word rules used by the weak labeller (see weak_labeler.py for how they combine).

Every rule has a NAME. The name is stored with each silver entity ("which rule produced this?"),
so we can later measure each rule's precision on the gold data and use it as its confidence.
"""
import re

# ---------------------------------------------------------------------------------------------
# 1. Text that is NOT part of the ingredient list: allergen statements, storage advice, claims.
#    A matching sentence is excluded from labelling (annotation guideline section 4).
# ---------------------------------------------------------------------------------------------
ALLERGENS = (r"milk|soy|soya|soybeans?|wheat|gluten|eggs?|nuts?|tree nuts|peanuts?|sesame|fish|shellfish|"
             r"crustaceans?|mustard|celery|lupin|sulphites|sulfites|almonds?|cashews?|hazelnuts?|walnuts?|"
             r"pecans?|pistachios?|molluscs|barley|oats|rye|dairy|cereals")
STATEMENT_START_RE = re.compile(
    r"\b(?:all[eè]rgen(?:s|ic)?(?:\s+(?:information|advice|statement|declaration))?|allergy\s+(?:advice|information)|"
    r"may\s+(?:also\s+)?contain|may\s+be\s+present|traces?\s+of|"
    rf"contains?\s*:?\s*(?=(?:{ALLERGENS})\b)|"
    r"(?:manufactured|made|produced|packed|processed|prepared)\s+(?:in|on)\s+(?:a\s+)?(?:factory|facility|site|premises|equipment|line)|"
    r"for\s+allergens|store\s+(?:in|at|below)|keep\s+(?:refrigerated|frozen|in|cool|dry)|best\s+before|"
    r"once\s+opened|nutrition(?:al)?\s+information|serving\s+suggestion|suitable\s+for|free\s+from|"
    r"not\s+suitable|consume\s+within|use\s+by|"
    r"no\s+(?:artificial|added|preservatives?|colou?rs?)|without\s+(?:artificial|added)|"
    r"typical\s+values|per\s+(?:100|serv)|energy\b|freephone|telephone|tel\b|careline|customer\s+care|"
    r"www\.|\.com\b|visit\s+us|find\s+out\s+more|plastic\s+bags?|suffocation|defrost|cooking\s+instructions|"
    r"distributed\s+by|marketed\s+by|manufactured\s+by|packed\s+by|imported\s+by|produce\s+of|product\s+of|"
    r"packaged\s+in|packed\s+in\s+a\s+protective)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------------------------
# 2. Framing words around an ingredient that are not part of the entity span.
# ---------------------------------------------------------------------------------------------
PCT = r"\d+(?:[.,]\d+)?\s?%"
FRAMING_PREFIX_RE = re.compile(
    r"^(?:"
    r"(?:and|or|with|plus|also|of|from|a|an|the|in|each)\s+"
    r"|ingredients?\b\s*"
    rf"|contains?\s+(?:less\s+than\s+)?(?:{PCT}\s+)?(?:or\s+less\s+)?(?:of\s+)?(?:each\s+of\s+)?(?:the\s+following\s*)?"
    rf"|(?:less\s+than|not\s+more\s+than|max(?:imum)?|min(?:imum)?)\s+{PCT}\s+(?:of\s+)?"
    rf"|{PCT}\s+or\s+less\s+(?:of\s+)?"
    r"|made\s+(?:with|from)\s+"
    r"|for\s+"
    r")",
    re.IGNORECASE,
)
TRAILING_NOISE_RE = re.compile(rf"(?:\s+{PCT}.*|\s+\d+(?:[.,]\d+)?\s?(?:g|mg|ml)\b.*|[\s*]+|"
                               r"\s+(?:of|from)\s+(?:vegetable|plant|animal|natural|non-animal)\s+origin.*|"
                               r"\s+in\s+(?:varying|variable)\s+proportions?.*|\s+in\s+addition\s+to.*|"
                               r"\s+(?:added\s+)?(?:to|for)\s+(?:protect|preserve|maintain|retain|promote|keep|prevent|"
                               r"freshness|quality|stability)\b.*|"
                               r"\s+for\s+(?:dusting|decoration|decorating|coating|glazing|frying|baking)\b.*|"
                               r"\s+(?:and|or|with|of|from|&|in)\s*)$", re.IGNORECASE)

# Pieces that are framing only -> no entity
SKIP_PIECE_RE = re.compile(
    r"^(?:ingredients?|contains?|and|or|of|the|following|organic|added|less\s+than|min|max|minimum|maximum|"
    r"each|approx|e|ins|no|one\s+or\s+more\s+of\s+the\s+following|in\s+varying\s+proportions|"
    r"varying\s+proportions|as\s+(?:a|an)\s+\w+|"
    r"(?:to|for)\s+(?:protect|preserve|maintain|retain|promote|help|keep|prevent|add|enhance|improve)\b.*|"
    r"contains?\s+one\s+or\s+more\s+of\s+the\s+following.*|.*\bor\s+less\s+of\b.*|"
    r".*\bcontains?\b.*|(?:this|these|it|we|our|us|i|you|your|my|please)\b.*|"   # sentences, not ingredients
    r"[\d\s.,%/+-]*)$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------------------------
# 3. Head-word rules: in English the LAST word of a noun phrase is its head,
#    so "refined palm oil" is an oil and "organic cane sugar" is a sugar.
# ---------------------------------------------------------------------------------------------
FAT_HEADS = {"oil", "oils", "fat", "fats", "shortening", "ghee", "lard", "margarine", "olein", "palmolein",
             "stearin", "tallow", "vanaspati"}
SUGAR_HEADS = {"sugar", "sugars", "syrup", "syrups", "dextrose", "fructose", "glucose", "sucrose", "honey",
               "molasses", "jaggery", "treacle"}
NOT_FAT_WORDS = {"peppermint", "spearmint", "mint", "lemon", "orange", "lime", "clove", "essential",
                 "eucalyptus", "aniseed", "bergamot", "grapefruit", "cinnamon", "wintergreen"}
NUT_BUTTER_WORDS = {"peanut", "almond", "nut", "cashew", "hazelnut", "apple", "seed", "pistachio", "cookie",
                    "pecan", "walnut"}
FLAVOUR_RE = re.compile(r"\bflavou?r(?:s|ing|ings|ed)?\b", re.IGNORECASE)
FLAVOUR_ENHANCER_RE = re.compile(r"\bflavou?r\s+enhancers?\b", re.IGNORECASE)

# Words a dictionary term may absorb in front of it when the term is an additive
# ("organic soy lecithin" -> one ADDITIVE span). Anything else in front is a separate entity.
ADDITIVE_MODIFIERS = {"organic", "natural", "pure", "purified", "refined", "food", "grade", "non-gmo", "gmo",
                      "free", "dried", "powdered", "liquid", "concentrated", "certified", "sustainable",
                      "naturally", "derived", "occurring", "e", "added"}
SUFFIX_MAX_PREFIX_WORDS = 3   # "partially inverted refiners syrup" is still one SUGAR entity
ADDITIVE_LABELS = {"ADDITIVE", "PRESERVATIVE", "COLOUR", "SWEETENER"}

# Connectors that may separate two entities inside one piece ("potassium benzoate and potassium sorbate")
CONNECTOR_RE = re.compile(r"\s*(?:\band\b|\bor\b|&|\band/or\b|/)\s*", re.IGNORECASE)
STOPWORDS = {"and", "or", "of", "with", "&", "from", "a", "an", "the", "in", "as", "for", "/", "-", "contains"}


def head_rule(words: list):
    """Apply head-word rules to a list of lower-cased words. Returns (label, rule_name) or None."""
    if not words:
        return None
    text = " ".join(words)
    if FLAVOUR_RE.search(text) and not FLAVOUR_ENHANCER_RE.search(text):
        return "FLAVOURING", "head:flavour_word"
    head = words[-1]
    if head in FAT_HEADS:
        if set(words) & NOT_FAT_WORDS:
            return "FLAVOURING", "head:flavour_oil"
        return "FAT", "head:fat_word"
    if head == "butter":
        if len(words) > 1 and words[-2] in NUT_BUTTER_WORDS:
            return "INGREDIENT", "head:nut_butter"
        return "FAT", "head:fat_word"
    if head in SUGAR_HEADS:
        return "SUGAR", "head:sugar_word"
    return None
