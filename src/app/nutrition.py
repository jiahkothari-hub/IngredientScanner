"""Nutrition facts for the daily-intake guidance: read them from the photo, or estimate them.

    from src.app.nutrition import parse_nutrition, estimate_from_similar_product
    parse_nutrition(ocr_text)    -> {"per_100g": {"sugars_g": 0.9, ...}, "serving_g": 25.0, "found": [...]}
    estimate_from_similar_product(ingredient_text) -> {"per_100g": {...}, "product_name": ..., "similarity": 0.83}

1. PARSING (preferred). Many photos include the nutrition table. Each line is searched for a nutrient
   keyword ("of which sugars", "saturated fat", "salt", "sodium" ...) followed by numbers with units.
   Tables usually show "per 100 g" first, then "per serving": we take the first number as per-100 g
   when a "100 g/ml" header is present. Sanity rules catch OCR slips: energy above 900 kcal/100 g is
   impossible, so it must be kJ (divided by 4.184); fat/sugars/... cannot exceed 100 g per 100 g;
   sodium (mg) is converted to salt (salt = sodium x 2.5).
2. ESTIMATE (fallback, clearly labelled). If the photo shows only the ingredient list, we look for the
   product in our Open Food Facts database (9,000 products with nutrition facts) whose ingredient list is
   most similar (TF-IDF cosine similarity >= 0.4) and offer its values as an estimate.
3. The user can always correct every number in the app.
"""
import re
from functools import lru_cache

import pandas as pd

from src.utils.config import project_path

NUTRIENTS = {           # key -> (display name, unit)
    "energy_kcal": ("Energy", "kcal"),
    "fat_g": ("Fat", "g"),
    "saturated_fat_g": ("Saturated fat", "g"),
    "trans_fat_g": ("Trans fat", "g"),
    "carbohydrate_g": ("Carbohydrate", "g"),
    "sugars_g": ("Sugars", "g"),
    "added_sugars_g": ("Added sugars", "g"),
    "fibre_g": ("Fibre", "g"),
    "protein_g": ("Protein", "g"),
    "salt_g": ("Salt", "g"),
}
# order matters: more specific patterns first ("saturated fat" before "fat", "sugars" before "carbohydrate")
PATTERNS = [
    ("trans_fat_g", r"trans\s*fat(?:ty\s*acids?)?|trans\b"),
    ("saturated_fat_g", r"satur\w*(?:\s*fat\w*)?(?:\s*acids?)?|saturates|sat\.?\s*fat"),
    ("added_sugars_g", r"(?:incl(?:udes)?\.?\s*)?added\s*sugars?"),
    ("sugars_g", r"(?:of\s*which\s*)?(?:total\s*)?sugars?"),
    ("fibre_g", r"(?:dietary\s*)?fib(?:re|er)"),
    ("protein_g", r"proteins?"),
    ("carbohydrate_g", r"(?:total\s*)?carbohydrates?|carbs"),
    ("salt_g", r"salt"),
    ("sodium_mg", r"sodium"),
    ("energy_kcal", r"energy(?!\s*from)|calories|kcal"),
    ("fat_g", r"(?:total\s*)?fat\b(?!\s*from)|lipids?"),
]
# "Serving size 25" when OCR lost the unit (assumed g), but not "Serving size 1 cup"
SERVING_NO_UNIT_RE = re.compile(r"serving\s*size\s*:?\s*(\d+(?:[.,]\d+)?)\b(?!\s*(?:cups?|pieces?|biscuits?|slices?|"
                                r"tbsp|tsp|pcs?|bars?|sachets?|packs?|cookies?|units?)\b)", re.IGNORECASE)
NUMBER_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(kcal|kj|mg|g|%)?", re.IGNORECASE)
SERVING_LINE_RE = re.compile(r"(?:serving|portion|serve)\s*(?:size)?[^\n]{0,30}?(\d+(?:[.,]\d+)?)\s*(g|ml)\b", re.IGNORECASE)


def _num(s: str) -> float:
    return float(s.replace(",", "."))


def parse_nutrition(text: str) -> dict:
    """Extract per-100 g values (and the serving size) from OCR text of a nutrition table."""
    per100 = {}
    found = []
    serving = None
    m = SERVING_LINE_RE.search(text) or SERVING_NO_UNIT_RE.search(text)
    if m:
        serving = _num(m.group(1))
    for line in text.split("\n"):
        low = line.lower()
        if re.search(r"energy\s*from\s*fat|calories\s*from\s*fat", low):
            continue
        for key, pattern in PATTERNS:
            km = re.search(r"(?<![a-z])(?:" + pattern + r")", low)
            if not km or key in per100:
                continue
            numbers = [(n.group(1), (n.group(2) or "").lower()) for n in NUMBER_RE.finditer(low[km.end():])]
            numbers = [(v, u) for v, u in numbers if u != "%"]          # skip "% daily value" columns
            if not numbers:
                continue
            value, unit = numbers[0]
            value = _num(value)
            if key == "energy_kcal":
                kcal = [_num(v) for v, u in numbers if u == "kcal"]
                kj = [_num(v) for v, u in numbers if u == "kj"]
                value = kcal[0] if kcal else (kj[0] / 4.184 if kj else value)
                if value > 900:                                       # impossible per 100 g -> it was kJ
                    value = value / 4.184
            elif key == "sodium_mg":
                value = value / 1000 if unit != "g" else value        # -> grams of sodium
                per100["salt_g"] = per100.get("salt_g") or round(value * 2.5, 2)
                found.append("sodium")
                break
            elif value > 100:                                         # grams per 100 g cannot exceed 100
                continue
            per100[key] = round(value, 2)
            found.append(key)
            break
    if per100.get("sugars_g") and per100.get("carbohydrate_g") and per100["sugars_g"] > per100["carbohydrate_g"]:
        per100.pop("sugars_g")                                       # OCR mix-up: sugars are part of carbs
    if per100.get("saturated_fat_g") and per100.get("fat_g") and per100["saturated_fat_g"] > per100["fat_g"]:
        per100.pop("saturated_fat_g")
    has_100 = bool(re.search(r"(?:per|/)\s*100\s*(?:g|ml)|100\s*(?:g|ml)", text, re.IGNORECASE))
    basis = "per 100 g"
    if per100 and not has_100 and serving:
        # US-style label: the values are per serving -> convert to per 100 g
        per100 = {k: round(v * 100 / serving, 2) for k, v in per100.items()}
        basis = f"converted from per serving ({serving:g} g/ml)"
    return {"per_100g": per100, "serving_g": serving, "has_per_100g_header": has_100, "found": found,
            "basis": basis, "source": "nutrition table in the photo" if per100 else ""}


# -------------------------------------------------------------------------------- estimate
@lru_cache(maxsize=1)
def _similarity_index():
    from sklearn.feature_extraction.text import TfidfVectorizer
    nutrition = pd.read_csv(project_path("data/knowledge_base/products_nutrition.csv"), dtype={"product_id": str})
    nutrition = nutrition.dropna(subset=["energy-kcal_100g", "sugars_100g", "fat_100g"])
    texts = pd.read_csv(project_path("data/processed/products.csv"), dtype={"product_id": str},
                        usecols=["product_id", "ingredients_text_raw"]).rename(columns={"ingredients_text_raw": "ingredients_text"})
    data = nutrition.merge(texts, on="product_id", how="inner").dropna(subset=["ingredients_text"])
    data = data.reset_index(drop=True)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    matrix = vectorizer.fit_transform(data["ingredients_text"].str.lower())
    return vectorizer, matrix, data


OFF_COLUMNS = {"energy-kcal_100g": "energy_kcal", "fat_100g": "fat_g", "saturated-fat_100g": "saturated_fat_g",
               "carbohydrates_100g": "carbohydrate_g", "sugars_100g": "sugars_g", "fiber_100g": "fibre_g",
               "proteins_100g": "protein_g", "salt_100g": "salt_g"}


def estimate_from_similar_product(ingredient_text: str, min_similarity: float = 0.4) -> dict:
    """Nutrition of the database product with the most similar ingredient list (or {} if none is close)."""
    if not ingredient_text or len(ingredient_text) < 15:
        return {}
    try:
        vectorizer, matrix, data = _similarity_index()
    except (FileNotFoundError, ValueError):
        return {}
    scores = (matrix @ vectorizer.transform([ingredient_text.lower()]).T).toarray().ravel()
    best = int(scores.argmax())
    if scores[best] < min_similarity:
        return {}
    row = data.iloc[best]
    per100 = {key: round(float(row[col]), 2) for col, key in OFF_COLUMNS.items() if pd.notna(row[col])}
    name = " - ".join(str(x) for x in (row["product_name"], row["brands"]) if pd.notna(x) and str(x))
    return {"per_100g": per100, "product_name": name or "unnamed product", "product_id": row["product_id"],
            "similarity": round(float(scores[best]), 2),
            "source": f"estimate: most similar product in the Open Food Facts database ({name})"}
