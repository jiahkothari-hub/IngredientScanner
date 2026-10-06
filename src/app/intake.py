"""Daily-intake guidance: is this portion a small or large part of a day's recommended limit?

    from src.app.intake import PROFILES, assess_portion, additive_guidance
    rows = assess_portion({"sugars_g": 56.3, "salt_g": 0.11, ...}, portion_g=15, profile=PROFILES["Adult (2000 kcal)"])
    adi = additive_guidance(result["entities"], body_weight_kg=60)

What the numbers are based on (official reference values, cited in the app):
  * WHO Guideline "Sugars intake for adults and children" (2015): free sugars < 10 % of energy
    (50 g/day at 2000 kcal), ideally < 5 % (25 g). ICMR-NIN Dietary Guidelines for Indians (2024) also
    recommend keeping added sugar below 5 % of energy.
  * WHO sodium guideline (2012): < 2 g sodium = < 5 g salt per day for adults, lower for children.
  * WHO saturated/trans fat guideline (2023): saturated fat < 10 % of energy, trans fat < 1 %.
  * EU Regulation 1169/2011, Annex XIII, Reference Intakes for an average adult: energy 2000 kcal,
    fat 70 g, saturates 20 g, carbohydrate 260 g, sugars 90 g, protein 50 g, salt 6 g.
  * EFSA (2010): fibre 25 g/day for adults.
  * UK FSA front-of-pack "traffic light" criteria per 100 g / 100 ml (low / medium / high).
  * Additives: acceptable daily intake (ADI) in mg per kg body weight from JECFA (FAO/WHO) or EFSA,
    see data/knowledge_base/additive_adi.csv.

Phrasing rule: these are reference values for the general healthy population, not personal medical
advice. The app says how a portion compares with them; it does not call a food "safe" or "dangerous".
"""
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from src.utils.config import project_path

TEASPOON_SUGAR_G = 4.0


@dataclass(frozen=True)
class Profile:
    name: str
    energy_kcal: float
    salt_g: float
    body_weight_kg: float


# energy needs and body weights are typical reference values (rounded); users can change them in the app
PROFILES = {
    "Adult (2000 kcal)": Profile("Adult (2000 kcal)", 2000, 5.0, 60),
    "Active adult (2500 kcal)": Profile("Active adult (2500 kcal)", 2500, 5.0, 70),
    "Teenager 11-17 (2200 kcal)": Profile("Teenager 11-17 (2200 kcal)", 2200, 5.0, 50),
    "Child 7-10 (1800 kcal)": Profile("Child 7-10 (1800 kcal)", 1800, 5.0, 28),
    "Child 4-6 (1400 kcal)": Profile("Child 4-6 (1400 kcal)", 1400, 3.0, 20),
}


def daily_limits(profile: Profile) -> dict:
    """nutrient -> (amount per day, kind, source). kind 'max' = stay below, 'target' = aim to reach."""
    e = profile.energy_kcal
    return {
        "sugars_g": (round(0.10 * e / 4), "max", "WHO 2015: free sugars < 10 % of energy "
                                                  f"(ideally < 5 % = {round(0.05 * e / 4)} g)"),
        "added_sugars_g": (round(0.10 * e / 4), "max", "WHO 2015: free sugars < 10 % of energy"),
        "fat_g": (round(0.30 * e / 9), "max", "WHO: total fat < 30 % of energy (EU reference intake 70 g)"),
        "saturated_fat_g": (round(0.10 * e / 9), "max", "WHO 2023: saturated fat < 10 % of energy (EU RI 20 g)"),
        "trans_fat_g": (round(0.01 * e / 9, 1), "max", "WHO 2023: trans fat < 1 % of energy"),
        "salt_g": (profile.salt_g, "max", "WHO 2012: salt < 5 g/day for adults (lower for young children)"),
        "energy_kcal": (e, "reference", "daily energy need of the selected profile (EU RI: 2000 kcal)"),
        "carbohydrate_g": (round(0.13 * e), "reference", "EU reference intake (260 g at 2000 kcal)"),
        "protein_g": (round(0.025 * e), "target", "EU reference intake (50 g at 2000 kcal)"),
        "fibre_g": (round(0.0125 * e), "target", "EFSA 2010: 25 g/day for adults"),
    }


# UK FSA front-of-pack criteria (per 100 g food / per 100 ml drink): (low <=, high >)
TRAFFIC_LIGHTS = {
    "food": {"fat_g": (3.0, 17.5), "saturated_fat_g": (1.5, 5.0), "sugars_g": (5.0, 22.5), "salt_g": (0.3, 1.5)},
    "drink": {"fat_g": (1.5, 8.75), "saturated_fat_g": (0.75, 2.5), "sugars_g": (2.5, 11.25), "salt_g": (0.3, 0.75)},
}
NAMES = {"energy_kcal": "Energy", "fat_g": "Fat", "saturated_fat_g": "Saturated fat", "trans_fat_g": "Trans fat",
         "carbohydrate_g": "Carbohydrate", "sugars_g": "Sugars", "added_sugars_g": "Added sugars",
         "fibre_g": "Fibre", "protein_g": "Protein", "salt_g": "Salt"}
UNITS = {"energy_kcal": "kcal"}


def traffic_light(key: str, per100: float, kind: str = "food") -> str:
    limits = TRAFFIC_LIGHTS[kind].get(key)
    if limits is None or per100 is None:
        return ""
    low, high = limits
    return "low" if per100 <= low else "high" if per100 > high else "medium"


def share_message(key: str, share: float) -> str:
    if share >= 1:
        return "this portion alone is more than the whole daily limit"
    if share >= 0.5:
        return "this portion uses more than half of the daily limit"
    if share >= 0.25:
        return "a large part of the daily limit - keep the rest of the day low in it"
    if share >= 0.10:
        return "a moderate part of the daily limit"
    return "a small part of the daily limit"


def assess_portion(per100: dict, portion_g: float, profile: Profile, kind: str = "food") -> list:
    """One row per nutrient with: amount in the portion, daily value, share, traffic light, and the
    amount of this product that alone would reach a 'max' limit."""
    rows = []
    limits = daily_limits(profile)
    for key in ["sugars_g", "added_sugars_g", "salt_g", "saturated_fat_g", "fat_g", "trans_fat_g", "energy_kcal",
                "carbohydrate_g", "protein_g", "fibre_g"]:
        value = per100.get(key)
        if value is None:
            continue
        daily, limit_kind, source = limits[key]
        amount = value * portion_g / 100
        share = amount / daily if daily else 0
        unit = UNITS.get(key, "g")
        max_product = round(daily / value * 100) if (limit_kind == "max" and value > 0) else None
        if limit_kind == "max":
            message = share_message(key, share)
        elif limit_kind == "target":
            message = f"{share:.0%} of the recommended daily amount (more is good)"
        else:
            message = f"{share:.0%} of a typical day's {NAMES[key].lower()}"
        extra = ""
        if key in ("sugars_g", "added_sugars_g") and amount > 0:
            extra = f" ≈ {amount / TEASPOON_SUGAR_G:.1f} teaspoons of sugar"
        rows.append({"key": key, "nutrient": NAMES[key], "per_100g": value, "in_portion": round(amount, 1),
                     "unit": unit, "daily_value": daily, "limit_kind": limit_kind, "share": share,
                     "traffic_light": traffic_light(key, value, kind), "max_product_g": max_product,
                     "message": message + extra, "source": source})
    return rows


def headline(rows: list, portion_g: float) -> tuple:
    """Overall sentence + level ('ok' / 'caution' / 'high') from the 'max' nutrients."""
    limited = [r for r in rows if r["limit_kind"] == "max"]
    if not limited:
        return "No nutrition values available for this product.", "unknown"
    worst = max(limited, key=lambda r: r["share"])
    reds = [r["nutrient"].lower() for r in limited if r["traffic_light"] == "high"]
    if worst["share"] >= 0.5:
        level = "high"
        text = (f"{portion_g:g} g of this product gives {worst['share']:.0%} of the daily limit for "
                f"{worst['nutrient'].lower()}. Have it occasionally or in a smaller portion.")
    elif worst["share"] >= 0.25 or reds:
        level = "caution"
        text = (f"{portion_g:g} g fits in a day, but uses {worst['share']:.0%} of the daily limit for "
                f"{worst['nutrient'].lower()}" + (f"; the product is high in {', '.join(reds)} per 100 g." if reds else "."))
    else:
        level = "ok"
        text = (f"{portion_g:g} g is a small part of the daily limits (highest: {worst['nutrient'].lower()}, "
                f"{worst['share']:.0%}).")
    return text, level


# -------------------------------------------------------------------------------- additives
@lru_cache(maxsize=1)
def adi_table() -> dict:
    """INS number -> ADI row."""
    path = project_path("data/knowledge_base/additive_adi.csv")
    table = {}
    for row in pd.read_csv(path, dtype=str).fillna("").to_dict("records"):
        for number in row["ins_numbers"].split("|"):
            table[number.strip().lower()] = row
    return table


def _adi_row(number: str):
    table = adi_table()
    number = (number or "").lower()
    for candidate in (number, number.rstrip("ivx"), "".join(ch for ch in number if ch.isdigit())):
        if candidate in table:
            return table[candidate]
    return None


def additive_guidance(entities: list, body_weight_kg: float) -> list:
    """One row per additive (by INS number) with its ADI scaled to the given body weight."""
    rows, seen = [], set()
    for e in entities:
        number = e.get("ins_number") or ""
        if not number:
            continue
        row = _adi_row(number)
        if row is None:
            continue
        key = row["substance"] if row["substance"] != "Low-concern additive group" else number
        if key in seen:
            continue
        seen.add(key)
        adi = float(row["adi_mg_per_kg"]) if row["adi_mg_per_kg"] else None
        name = e.get("canonical_name") or e["text"]
        if adi is not None:
            per_day = adi * body_weight_kg
            limit = (f"{adi:g} mg per kg body weight per day → about {per_day:,.0f} mg/day for {body_weight_kg:g} kg"
                     + (f" ({row['adi_basis']})" if row["adi_basis"] else ""))
        elif row["substance"] == "Low-concern additive group":
            limit = "No numerical limit needed (ADI 'not specified')"
        else:
            limit = "No numerical ADI"
        rows.append({"additive": f"{name} (INS {number})", "group": row["substance"], "adi_mg_per_kg": adi,
                     "daily_limit": limit, "authority": row["authority"], "note": row["note"]})
    return rows


SPECIAL_SUBSTANCES = [
    # (words to look for in the ingredient list, title, guidance)
    (("caffeine", "guarana", "coffee"), "Caffeine",
     "EFSA (2015): up to 400 mg per day is of no safety concern for healthy adults (200 mg in pregnancy), "
     "and about 3 mg per kg body weight for children and teenagers. A 250 ml energy drink usually has "
     "about 80 mg, a 330 ml cola about 30-35 mg (check the label)."),
    (("aspartame", "phenylalanine"), "Phenylalanine (from aspartame)",
     "Not suitable for people with phenylketonuria (PKU). For everyone else the ADI below applies."),
    (("sorbitol", "maltitol", "xylitol", "isomalt", "mannitol", "lactitol", "erythritol", "polyol"), "Polyols",
     "Sugar alcohols: eating large amounts at once (roughly 20 g or more, depending on the person) can have "
     "a laxative effect."),
    (("partially hydrogenated", "hydrogenated vegetable", "vanaspati"), "Possible trans fats",
     "Partially hydrogenated fats/vanaspati can contain trans fats. WHO: keep trans fat below 1 % of energy "
     "(about 2 g/day at 2000 kcal); FSSAI limits trans fat in fats and oils to 2 %."),
]


def special_notes(text: str) -> list:
    low = text.lower()
    return [{"substance": title, "note": note} for words, title, note in SPECIAL_SUBSTANCES
            if any(w in low for w in words)]
