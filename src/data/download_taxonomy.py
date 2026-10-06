"""Download and parse Open Food Facts' additive taxonomy (a reliable public reference list).

Usage (from the project root):
    python -m src.data.download_taxonomy

Why: we need a trustworthy mapping such as
    E330 / INS 330  ->  "Citric acid"  ->  functional classes [acidity regulator, antioxidant, ...]
Typing this by hand would be slow and error-prone. Open Food Facts maintains it as plain-text
"taxonomy" files in their GitHub repository (ODbL licence). We pin a specific commit so every
team member gets exactly the same file.

Note: INS numbers (Codex Alimentarius, used on Indian labels) and E-numbers (EU) use the same
numbering, e.g. INS 330 = E330 = citric acid. So one table keyed by the number serves both.

Outputs:
    data/raw/off_taxonomy/additives.txt, additives_classes.txt   (unchanged source files)
    data/processed/additives_reference.csv     one row per additive number
    data/processed/function_classes.csv        one row per functional class (preservative, colour, ...)
"""
import re
from pathlib import Path

import pandas as pd
import requests

from src.utils.config import project_path

OFF_COMMIT = "81288a7f46b40c7ecf371865358afb898422196a"  # openfoodfacts-server main, 2026-10-01
BASE_URL = f"https://raw.githubusercontent.com/openfoodfacts/openfoodfacts-server/{OFF_COMMIT}/taxonomies/"
FILES = ["additives.txt", "additives_classes.txt"]

RAW_DIR = project_path("data/raw/off_taxonomy")
ADDITIVES_CSV = project_path("data/processed/additives_reference.csv")
CLASSES_CSV = project_path("data/processed/function_classes.csv")


def parse_taxonomy(path: Path) -> list:
    """Parse an OFF taxonomy file into a list of entries.

    The file is made of blocks separated by blank lines. Inside a block:
        < en:parent             -> parent entry (ignored here)
        en: Name, Synonym, ...  -> English names (the first one is the canonical name)
        xx: Name, ...           -> language-independent names (e.g. "E330")
        fr: Nom, ...            -> other languages (ignored: we work on English text)
        key:en: value           -> a property, e.g. "e_number:en: 330"
        # ...                   -> comment
    Returns [{"names_en": [...], "properties": {key: value}}, ...]
    """
    entries = []
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")):
        names, properties = [], {}
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("<"):
                continue
            prop = re.match(r"^([a-z0-9_]+):([a-z]{2}):\s*(.*)$", line)
            if prop and prop.group(2) == "en":           # property line, e.g. e_number:en: 330
                properties[prop.group(1)] = prop.group(3)
                continue
            lang = re.match(r"^(en|xx):\s*(.*)$", line)  # name line
            if lang:
                # names are comma-separated; "\," is an escaped comma inside a name
                names += [n.strip().replace("\\,", ",") for n in re.split(r"(?<!\\),", lang.group(2)) if n.strip()]
        if names:
            entries.append({"names_en": names, "properties": properties})
    return entries


# "E330", "E 330", "E150d", "E322(i)", "E500(ii)" -> groups: digits+letter, roman sub-number
CODE_NAME_RE = re.compile(r"^E\s?(\d{3,4}[a-z]?)\s*(?:\(([ivx]+)\))?$", re.IGNORECASE)


def additive_number(entry: dict):
    """The additive's number in Open Food Facts tag form: '330', '150d', '322i', '500ii'.

    Usually stored as the property 'e_number', but some entries (e.g. E270) only have it
    as their first name, and sub-types are written 'E322(i)' while tags say 'e322i'.
    """
    for name in entry["names_en"]:
        match = CODE_NAME_RE.match(name)
        if match:
            return (match.group(1) + (match.group(2) or "")).lower()
    number = entry["properties"].get("e_number")
    return number.lower().replace(" ", "") if number else None


def build_additives_table(entries: list) -> pd.DataFrame:
    rows = []
    for entry in entries:
        number = additive_number(entry)
        if not number:
            continue
        # canonical name = first English name that is not just a code ("E330", "E322(i)")
        names = [n for n in entry["names_en"] if not CODE_NAME_RE.match(n)]
        classes = re.findall(r"en:([\w-]+)", entry["properties"].get("additives_classes", ""))
        rows.append({
            "number": number,                          # "330", "150d", "322i"
            "e_code": "E" + number,                    # "E330"
            "name": names[0] if names else "",
            "synonyms": "|".join(dict.fromkeys(names[1:])),  # de-duplicated, order kept
            "function_classes": "|".join(classes),     # "antioxidant|sequestrant"
        })
    table = pd.DataFrame(rows).drop_duplicates(subset="number").sort_values("e_code").reset_index(drop=True)

    # Sub-types (322i, 500ii, 150d) often have no classes of their own: inherit from the parent (322, 500, 150)
    by_number = dict(zip(table["number"], table["function_classes"]))
    parent = table["number"].str.extract(r"^(\d+)")[0]
    inherit = (table["function_classes"] == "") & (parent != table["number"])
    table["classes_source"] = "taxonomy"
    table.loc[inherit, "function_classes"] = parent[inherit].map(by_number).fillna("")
    table.loc[inherit & (table["function_classes"] != ""), "classes_source"] = "inherited_from_parent"
    table.loc[table["function_classes"] == "", "classes_source"] = "missing"
    return table


def build_classes_table(entries: list) -> pd.DataFrame:
    rows = []
    for entry in entries:
        canonical = entry["names_en"][0]
        rows.append({
            "class_id": re.sub(r"[^a-z0-9]+", "-", canonical.lower()).strip("-"),
            "name": canonical,
            "synonyms": "|".join(dict.fromkeys(entry["names_en"][1:])),
        })
    return pd.DataFrame(rows)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        response = requests.get(BASE_URL + name, timeout=60)
        response.raise_for_status()
        (RAW_DIR / name).write_text(response.text, encoding="utf-8")
        print(f"Downloaded {name} ({len(response.text) / 1e3:,.0f} kB)")

    additives = build_additives_table(parse_taxonomy(RAW_DIR / "additives.txt"))
    classes = build_classes_table(parse_taxonomy(RAW_DIR / "additives_classes.txt"))
    additives.to_csv(ADDITIVES_CSV, index=False)
    classes.to_csv(CLASSES_CSV, index=False)
    print(f"Saved {len(additives)} additives -> {ADDITIVES_CSV.relative_to(project_path(''))}")
    print(f"Saved {len(classes)} functional classes -> {CLASSES_CSV.relative_to(project_path(''))}")


if __name__ == "__main__":
    main()
