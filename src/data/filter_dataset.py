"""Build the manageable project dataset from the full Open Food Facts export.

Usage (from the project root, after download_dataset.py):
    python -m src.data.filter_dataset

What it does, in one pass over the ~4M-row export:
  1. Reads only the ~10 columns listed in configs/data_config.yaml (chunk by chunk,
     so memory stays small).
  2. Profiles the WHOLE source: number of products, missing values per column,
     country distribution  -> reports/source_profile.json
  3. Keeps products sold in the configured English-speaking country groups.

Then, on those candidate products:
  4. Removes empty / too short / too long ingredient texts.
  5. Keeps English-language lists (see src/data/language.py).
  6. Removes duplicate barcodes and exact-duplicate ingredient lists.
  7. Samples up to a fixed quota per country group (fixed random seed).
  -> data/processed/products.csv  and  reports/filter_log.json (the "funnel")

The original ingredient text is stored unchanged in `ingredients_text_raw`.
Nothing in this script edits it; normalisation happens later in src/preprocessing/.
"""
import json
import re

import pandas as pd

from src.data.language import detect_language
from src.utils.config import load_config, project_path

CHUNK_ROWS = 200_000
# Open Food Facts stores some missing values as literal strings (36k+ lists are just "undefined").
PLACEHOLDER_VALUES = {"undefined", "null", "none", "nan", "n/a", "na", "-", "?", "unknown"}
SOURCE_LANGUAGE_SAMPLE = 0.02  # fraction of source rows used to estimate the global language mix


def read_export_in_chunks(path, columns):
    """Yield DataFrames of the OFF export. All values are strings; empty cells are NaN.

    The export is tab-separated with NO quoting (quotes inside text are literal),
    hence quoting=3 (csv.QUOTE_NONE). A handful of malformed lines are skipped.
    """
    return pd.read_csv(
        path, sep="\t", quoting=3, usecols=columns, dtype=str, chunksize=CHUNK_ROWS,
        compression="gzip", encoding="utf-8", encoding_errors="replace",
        on_bad_lines="skip", keep_default_na=False, na_values=[""],
    )


def assign_country_group(countries: str, country_groups: dict):
    """Return the first matching group name (groups are checked in config order), else None.

    Products sold in several countries get the FIRST group in the config, so a product
    sold in both India and the UK counts as 'india' (India is listed first on purpose:
    Indian labels are the main target style).
    """
    if not isinstance(countries, str):
        return None
    product_countries = {c.strip() for c in countries.split(",")}
    for group, members in country_groups.items():
        if product_countries.intersection(members):
            return group
    return None


def is_placeholder(value) -> bool:
    """True for missing values and placeholder strings such as 'undefined'."""
    return not isinstance(value, str) or value.strip().lower() in PLACEHOLDER_VALUES | {""}


def dedup_key(text: str) -> str:
    """Key for exact-duplicate detection: lowercase, single spaces, no space before punctuation."""
    text = re.sub(r"\s+", " ", text.lower()).strip()
    return re.sub(r"\s+([,.;:)\]])", r"\1", text)


def profile_and_collect_candidates(config):
    columns = config["columns"]
    groups = config["filter"]["country_groups"]
    raw_path = project_path(config["source"]["raw_file"])
    if not raw_path.exists():
        raise SystemExit(f"{raw_path} not found. Run `python -m src.data.download_dataset` first.")

    total_rows = 0
    non_missing = pd.Series(0, index=columns)
    country_counts = pd.Series(dtype="int64")
    language_counts = pd.Series(dtype="int64")
    candidates = []

    for i, chunk in enumerate(read_export_in_chunks(raw_path, columns)):
        total_rows += len(chunk)
        # a cell counts as present only if it is non-empty after stripping whitespace
        # (we only LOOK at stripped values here; the data itself is not modified)
        present = chunk.apply(lambda col: ~col.map(is_placeholder))
        non_missing += present.sum()

        countries = chunk["countries_en"].dropna().str.split(",").explode().str.strip()
        country_counts = country_counts.add(countries.value_counts(), fill_value=0)

        # Language mix of the WHOLE source, estimated on a seeded 2% sample (it is slow to do all rows)
        with_text = chunk["ingredients_text"][~chunk["ingredients_text"].map(is_placeholder)]
        sample = with_text.sample(frac=SOURCE_LANGUAGE_SAMPLE, random_state=config["random_seed"] + i)
        langs = sample.map(lambda t: detect_language(t)[0])
        language_counts = language_counts.add(langs.value_counts(), fill_value=0)

        chunk["country_group"] = chunk["countries_en"].map(lambda c: assign_country_group(c, groups))
        candidates.append(chunk[chunk["country_group"].notna()])
        print(f"  read {total_rows:,} rows", flush=True)

    profile = {
        "source_file": str(config["source"]["raw_file"]),
        "total_products": int(total_rows),
        "missing_percent": {c: round(100 * (1 - non_missing[c] / total_rows), 2) for c in columns},
        "top_countries": {k: int(v) for k, v in country_counts.sort_values(ascending=False).head(25).items()},
        "language_estimate_from_sample": {
            "sample_fraction_of_rows_with_ingredients": SOURCE_LANGUAGE_SAMPLE,
            "note": "only en/fr/de/es/it/nl/pt are detected; every other language counts as 'unknown'",
            "counts": {k: int(v) for k, v in language_counts.sort_values(ascending=False).items()},
        },
    }
    return profile, pd.concat(candidates, ignore_index=True)


def filter_candidates(df: pd.DataFrame, config) -> tuple:
    """Apply the filters one by one and record how many products survive each step."""
    fcfg = config["filter"]
    funnel = [("in English-speaking country groups", len(df))]

    df = df[df["code"].notna()]
    df = df.drop_duplicates(subset="code", keep="first")
    funnel.append(("unique barcode", len(df)))

    df = df[~df["ingredients_text"].map(is_placeholder)].copy()
    funnel.append(("has ingredients_text (not empty / 'undefined')", len(df)))

    n_chars = df["ingredients_text"].str.strip().str.len()
    df = df[(n_chars >= fcfg["min_chars"]) & (n_chars <= fcfg["max_chars"])].copy()
    funnel.append((f"length {fcfg['min_chars']}-{fcfg['max_chars']} chars", len(df)))

    detected = df["ingredients_text"].map(detect_language)
    df["detected_language"] = detected.map(lambda x: x[0])
    df["english_score"] = detected.map(lambda x: round(x[1], 3))
    language_mix = df["detected_language"].value_counts().to_dict()
    # English if (a) English marker words clearly dominate (bilingual EN/FR labels score ~0.5-0.9
    # and are removed), or (b) no marker word at all but plain ASCII text from an English-speaking
    # market, e.g. "Bing cherries." -> short single-ingredient lists are not lost.
    by_markers = (df["detected_language"] == "en") & (df["english_score"] >= fcfg["min_english_score"])
    assumed = (df["detected_language"] == "unknown") & df["ingredients_text"].map(str.isascii)
    df["english_rule"] = "marker_words"
    df.loc[assumed, "english_rule"] = "assumed_ascii_no_markers"
    df = df[by_markers | assumed]
    funnel.append(("English ingredient list", len(df)))

    df = df.assign(_key=df["ingredients_text"].map(dedup_key))
    duplicate_counts = df["_key"].value_counts()
    df = df.drop_duplicates(subset="_key", keep="first").copy()
    df["n_products_with_same_text"] = df["_key"].map(duplicate_counts)
    funnel.append(("unique ingredient list (exact duplicates removed)", len(df)))

    sampled = []
    for group, quota in fcfg["group_quota"].items():
        part = df[df["country_group"] == group]
        if quota is not None and len(part) > quota:
            part = part.sample(n=quota, random_state=config["random_seed"])
        sampled.append(part)
    df = pd.concat(sampled).sort_values("code").drop(columns="_key")
    funnel.append(("after per-country-group sampling", len(df)))

    log = {
        "funnel": [{"step": s, "products": int(n)} for s, n in funnel],
        "detected_language_among_length_filtered": {k: int(v) for k, v in language_mix.items()},
        "final_country_groups": {k: int(v) for k, v in df["country_group"].value_counts().items()},
        "final_english_rule": {k: int(v) for k, v in df["english_rule"].value_counts().items()},
    }
    return df, log


def to_project_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Rename OFF columns to the names used in the rest of the project."""
    out = pd.DataFrame({
        "product_id": df["code"],
        "product_name": df["product_name"],
        "brands": df["brands"],
        "country_group": df["country_group"],
        "countries": df["countries_en"],
        "main_category": df["main_category_en"],
        "categories": df["categories_en"],
        "ingredients_text_raw": df["ingredients_text"],   # original text, never modified
        "n_chars": df["ingredients_text"].str.len(),
        "detected_language": df["detected_language"],
        "english_score": df["english_score"],
        "english_rule": df["english_rule"],
        "n_products_with_same_text": df["n_products_with_same_text"],
        "off_additives_n": df["additives_n"],
        "off_additives_tags": df["additives_tags"],       # OFF's own detection, for EDA only
    })
    return out.reset_index(drop=True)


def main():
    config = load_config()
    out = config["outputs"]

    print("Pass over the full export (profiling + collecting candidates)...")
    profile, candidates = profile_and_collect_candidates(config)
    project_path(out["source_profile"]).write_text(json.dumps(profile, indent=2), encoding="utf-8")
    candidates.to_csv(project_path(out["candidates"]), index=False, compression="gzip")
    print(f"Source: {profile['total_products']:,} products; candidates: {len(candidates):,}")

    products, log = filter_candidates(candidates, config)
    project_path(out["filter_log"]).write_text(json.dumps(log, indent=2), encoding="utf-8")
    to_project_schema(products).to_csv(project_path(out["products"]), index=False)

    for step in log["funnel"]:
        print(f"  {step['products']:>9,}  {step['step']}")
    print(f"Saved {len(products):,} products to {out['products']}")


if __name__ == "__main__":
    main()
