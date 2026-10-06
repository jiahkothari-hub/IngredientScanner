"""Build the knowledge base (KB) used for entity linking and for the app's explanations.

Usage:  python -m src.kb.build_knowledge_base [--skip-nutrition]

Tables written to data/knowledge_base/:
  additives.csv          one row per INS/E number: name, synonyms, NER label, functional classes,
                         class descriptions, vegetarian/vegan flags, reference links (Wikidata, EFSA)
  function_classes.csv   functional classes with their official-style definition
                         ("Emulsifiers are substances which make it possible to ...")
  ingredient_terms.csv   every dictionary term (sugars, sweeteners, fats, flavourings, additives,
                         class names) with its category - used for exact and fuzzy linking
  categories.csv         plain-language description of each NER category
  products_nutrition.csv nutrition facts per 100 g for the 22k project products (from the OFF export)

Every text is descriptive (what something is / what it is used for). The KB contains NO health
ratings or risk scores on purpose: the project identifies and interprets ingredients, it does not
judge them.
"""
import argparse
import re

import pandas as pd

from src.data.download_taxonomy import parse_taxonomy
from src.labeling.lexicon import label_for_number, load_lexicon
from src.utils.config import load_config, project_path

KB_DIR = project_path("data/knowledge_base")
TAXONOMY_DIR = project_path("data/raw/off_taxonomy")
NUTRITION_COLUMNS = ["energy-kcal_100g", "fat_100g", "saturated-fat_100g", "carbohydrates_100g", "sugars_100g",
                     "fiber_100g", "proteins_100g", "salt_100g"]

CATEGORY_DESCRIPTIONS = {
    "SUGAR": "A sugar or syrup added to the food. Labels use many different names for sugars "
             "(e.g. dextrose, glucose syrup, invert syrup, maltose).",
    "SWEETENER": "A sweetener other than sugar: an intense sweetener (e.g. sucralose) or a polyol / sugar alcohol "
                 "(e.g. sorbitol).",
    "FAT": "An oil or fat, e.g. palm oil, palmolein, vegetable fat, butter or ghee.",
    "PRESERVATIVE": "A food additive whose main function is preservation: it helps protect food against "
                    "spoilage by micro-organisms.",
    "COLOUR": "A food additive that adds or restores colour.",
    "ADDITIVE": "A food additive with an INS/E number. Its function is shown where the label or the reference "
                "data states it.",
    "INS_CODE": "An additive code. INS (International Numbering System, Codex Alimentarius) and E-numbers (EU) "
                "use the same numbers, e.g. INS 330 = E330 = citric acid.",
    "FUNCTION_CLASS": "The functional class written on the label (e.g. 'acidity regulator', 'emulsifier'). "
                      "It tells what job the additive next to it does.",
    "FLAVOURING": "A flavouring: natural, nature-identical or artificial flavouring substances.",
    "INGREDIENT": "A food ingredient (e.g. wheat flour, milk solids, salt, spices).",
}


def class_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def build_function_classes() -> pd.DataFrame:
    rows = []
    for entry in parse_taxonomy(TAXONOMY_DIR / "additives_classes.txt"):
        name = entry["names_en"][0]
        rows.append({"class_id": class_id(name), "name": name,
                     "synonyms": "|".join(dict.fromkeys(entry["names_en"][1:])),
                     "description": entry["properties"].get("description", ""),
                     "wikidata": entry["properties"].get("wikidata", "")})
    return pd.DataFrame(rows).drop_duplicates("class_id")


def build_additives(classes: pd.DataFrame) -> pd.DataFrame:
    reference = pd.read_csv(project_path("data/processed/additives_reference.csv"), dtype=str).fillna("")
    # extra properties from the raw taxonomy, keyed by the same number
    props = {}
    for entry in parse_taxonomy(TAXONOMY_DIR / "additives.txt"):
        number = entry["properties"].get("e_number", "")
        for name in entry["names_en"]:
            m = re.match(r"^E\s?(\d{3,4}[a-z]?)\s*(?:\(([ivx]+)\))?$", name, re.IGNORECASE)
            if m:
                number = (m.group(1) + (m.group(2) or "")).lower()
                break
        if number:
            props.setdefault(number.lower(), entry["properties"])
    class_names = dict(zip(classes["class_id"], classes["name"]))
    class_desc = dict(zip(classes["class_id"], classes["description"]))

    rows = []
    for r in reference.itertuples():
        p = props.get(r.number, {})
        fclasses = [c for c in r.function_classes.split("|") if c]
        wikidata = p.get("wikidata", "")
        rows.append({
            "kb_id": f"INS:{r.number}",
            "ins_number": r.number,
            "e_code": r.e_code,
            "name": r.name,
            "synonyms": r.synonyms,
            "ner_label": label_for_number(r.number),
            "function_classes": "|".join(fclasses),
            "function_classes_text": ", ".join(class_names.get(c, c.replace("-", " ")) for c in fclasses),
            "function_description": " ".join(class_desc.get(c, "") for c in fclasses[:1]),
            "vegetarian": p.get("vegetarian", ""),
            "vegan": p.get("vegan", ""),
            "wikidata_url": f"https://www.wikidata.org/wiki/{wikidata}" if wikidata.startswith("Q") else "",
            "efsa_evaluation_url": p.get("efsa_evaluation_url", ""),
            "source": "Open Food Facts additives taxonomy (ODbL)",
        })
    return pd.DataFrame(rows)


def build_terms() -> pd.DataFrame:
    rows = [{"term": e.term, "category": e.label, "ins_number": e.number, "class_id": e.class_id,
             "source": e.source} for e in load_lexicon().values()]
    return pd.DataFrame(rows).sort_values(["category", "term"])


def build_nutrition() -> pd.DataFrame:
    config = load_config()
    products = pd.read_csv(project_path(config["outputs"]["products"]), dtype={"product_id": str})
    wanted = set(products["product_id"])
    raw = project_path(config["source"]["raw_file"])
    parts = []
    for chunk in pd.read_csv(raw, sep="\t", quoting=3, usecols=["code"] + NUTRITION_COLUMNS, dtype={"code": str},
                             chunksize=200_000, compression="gzip", on_bad_lines="skip", low_memory=False):
        parts.append(chunk[chunk["code"].isin(wanted)])
    nutrition = pd.concat(parts).drop_duplicates("code").rename(columns={"code": "product_id"})
    for col in NUTRITION_COLUMNS:
        nutrition[col] = pd.to_numeric(nutrition[col], errors="coerce").round(2)
    meta = products[["product_id", "product_name", "brands", "country_group", "main_category"]]
    return meta.merge(nutrition, on="product_id", how="left")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-nutrition", action="store_true", help="skip the slow pass over the raw export")
    args = parser.parse_args()
    KB_DIR.mkdir(parents=True, exist_ok=True)
    classes = build_function_classes()
    classes.to_csv(KB_DIR / "function_classes.csv", index=False)
    additives = build_additives(classes)
    additives.to_csv(KB_DIR / "additives.csv", index=False)
    build_terms().to_csv(KB_DIR / "ingredient_terms.csv", index=False)
    pd.DataFrame([{"category": k, "description": v} for k, v in CATEGORY_DESCRIPTIONS.items()]).to_csv(
        KB_DIR / "categories.csv", index=False)
    print(f"additives: {len(additives)}, function classes: {len(classes)} "
          f"({(classes['description'] != '').sum()} with description)")
    if not args.skip_nutrition and project_path(load_config()["source"]["raw_file"]).exists():
        nutrition = build_nutrition()
        nutrition.to_csv(KB_DIR / "products_nutrition.csv", index=False)
        print(f"products with nutrition: {nutrition['energy-kcal_100g'].notna().sum():,} / {len(nutrition):,}")


if __name__ == "__main__":
    main()
