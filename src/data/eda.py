"""Exploratory data analysis of the project dataset.

Usage (from the project root, after filter_dataset.py and download_taxonomy.py):
    python -m src.data.eda

Inputs:  reports/source_profile.json, reports/filter_log.json,
         data/processed/products.csv, data/processed/additives_reference.csv
Outputs: reports/figures/*.png, reports/dataset_statistics.json, reports/dataset_statistics.md

Note: this stage runs BEFORE the real preprocessing/weak labelling. Ingredient "segments" here are
a crude split on commas/semicolons/brackets, and additive counts come from Open Food Facts' own
`additives_tags`. The figure of our own entity-label distribution is produced later from the silver
labels (Stage 3), so the two can be compared.
"""
import json
import re
from collections import Counter

import matplotlib

matplotlib.use("Agg")  # write files, no window needed
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import load_config, project_path

FIG_DIR = project_path("reports/figures")

# One consistent, colour-blind-checked palette (categorical order is fixed, never cycled).
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GREY, INK, MUTED = "#c9c7c0", "#0b0b0b", "#52514e"
GROUP_ORDER = ["india", "uk_ireland", "north_america", "other_english"]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
})

# --- Regular expressions used only for this exploratory count -------------------------------
# E-number style: "E330", "E 330", "e-330", "E150d"
E_CODE_RE = re.compile(r"\bE\s?-?\d{3,4}[a-z]?\b", re.IGNORECASE)
# INS style: "INS 330", "INS-330", "INS No. 330", "INS:330"
INS_CODE_RE = re.compile(r"\bINS\s*(?:No\.?|Number)?\s*[:.\-]?\s*\d{3,4}", re.IGNORECASE)
# Bare number in brackets after a class name, as on many Indian labels: "Acidity Regulator (330)"
BARE_CODE_RE = re.compile(r"\(\s*\d{3,4}\s*[a-z]?\s*(?:\(\s*[ivx]+\s*\))?\s*\)", re.IGNORECASE)
PERCENT_RE = re.compile(r"\d+(?:[.,]\d+)?\s*%")
SEGMENT_SPLIT_RE = re.compile(r"[,;()\[\]{}]|\.\s")


def crude_segments(text: str) -> list:
    """Split an ingredient list into rough pieces for counting (NOT the real preprocessing)."""
    pieces = []
    for piece in SEGMENT_SPLIT_RE.split(str(text).lower()):
        piece = PERCENT_RE.sub("", piece)
        piece = re.sub(r"^\W+|\W+$", "", re.sub(r"\s+", " ", piece)).strip()
        if piece and not piece.isdigit():
            pieces.append(piece)
    return pieces


def code_style(text: str) -> str:
    """Which notation does the label use for additive codes? (first match in priority order)"""
    if INS_CODE_RE.search(text):
        return "INS number (INS 330)"
    if E_CODE_RE.search(text):
        return "E-number (E330)"
    if BARE_CODE_RE.search(text):
        return "bare number (330)"
    return "no code"


def save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved reports/figures/{name}")


def label_bars(ax, values, fmt="{:,.0f}", pad=0.01):
    xmax = ax.get_xlim()[1]
    for i, v in enumerate(values):
        ax.text(v + pad * xmax, i, fmt.format(v), va="center", color=MUTED, fontsize=8)


# --- Figures --------------------------------------------------------------------------------

def fig_dataset_size(profile, filter_log):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})

    funnel = filter_log["funnel"]
    steps = ["Full Open Food Facts export"] + [s["step"] for s in funnel]
    counts = [profile["total_products"]] + [s["products"] for s in funnel]
    ax1.barh(range(len(steps)), counts, color=BLUE, height=0.6)
    ax1.set_yticks(range(len(steps)), steps)
    ax1.invert_yaxis()
    ax1.set_xscale("log")
    ax1.set_xlabel("products (log scale)")
    ax1.set_title("From the full export to the project dataset", loc="left", color=INK)
    for i, v in enumerate(counts):
        ax1.text(v * 1.08, i, f"{v:,}", va="center", color=MUTED, fontsize=8)
    ax1.set_xlim(right=max(counts) * 6)

    missing = pd.Series(profile["missing_percent"]).sort_values()
    ax2.barh(missing.index, missing.values, color=BLUE, height=0.6)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("% of ALL products with the field empty")
    ax2.set_title("Missing values in the full export", loc="left", color=INK)
    label_bars(ax2, missing.values, "{:.1f}%")
    save(fig, "01_dataset_size_and_missingness.png")


def fig_length_distribution(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.4))
    ax1.hist(df["n_chars"], bins=60, color=BLUE, edgecolor="white", linewidth=0.5)
    ax1.axvline(df["n_chars"].median(), color=INK, linewidth=1, linestyle="--")
    ax1.text(df["n_chars"].median(), ax1.get_ylim()[1] * 0.92, f"  median {df['n_chars'].median():.0f}",
             color=INK, fontsize=8)
    ax1.set_xlabel("characters in ingredient list")
    ax1.set_ylabel("products")
    ax1.set_title("Ingredient-list length (characters)", loc="left", color=INK)

    clipped = df["n_segments"].clip(upper=60)
    ax2.hist(clipped, bins=range(0, 62, 2), color=BLUE, edgecolor="white", linewidth=0.5)
    ax2.axvline(df["n_segments"].median(), color=INK, linewidth=1, linestyle="--")
    ax2.text(df["n_segments"].median(), ax2.get_ylim()[1] * 0.92, f"  median {df['n_segments'].median():.0f}",
             color=INK, fontsize=8)
    ax2.set_xlabel("ingredient segments (crude split; 60 = 60 or more)")
    ax2.set_title("Ingredient-list length (segments)", loc="left", color=INK)
    save(fig, "02_ingredient_length_distribution.png")


def fig_top_terms(segment_counts, n_products, top_n=30):
    top = segment_counts.most_common(top_n)
    names = [t for t, _ in top]
    pct = [100 * c / n_products for _, c in top]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(names, pct, color=BLUE, height=0.65)
    ax.invert_yaxis()
    ax.set_xlabel("% of products whose list contains the segment")
    ax.set_title(f"Top {top_n} ingredient segments", loc="left", color=INK)
    label_bars(ax, pct, "{:.1f}%")
    save(fig, "03_top_ingredient_terms.png")


def fig_additive_codes(top_additives, style_table, n_products):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1, 1.1]})

    labels = [f"{code}  {name}"[:38] for code, name, _ in top_additives]
    pct = [100 * c / n_products for _, _, c in top_additives]
    ax1.barh(labels, pct, color=BLUE, height=0.65)
    ax1.invert_yaxis()
    ax1.set_xlabel("% of products (Open Food Facts additive detection)")
    ax1.set_title("Most frequent additives (INS/E numbers)", loc="left", color=INK)
    label_bars(ax1, pct, "{:.1f}%")

    styles = ["INS number (INS 330)", "E-number (E330)", "bare number (330)", "no code"]
    colors = [ORANGE, BLUE, AQUA, GREY]
    left = pd.Series(0.0, index=style_table.index)
    for style, color in zip(styles, colors):
        values = style_table.get(style, pd.Series(0.0, index=style_table.index))
        ax2.barh(style_table.index, values, left=left, color=color, label=style,
                 height=0.6, edgecolor="white", linewidth=2)
        for i, (v, l) in enumerate(zip(values, left)):
            if v >= 6:
                ax2.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8,
                         color="white" if style != "no code" else INK)
        left += values
    ax2.invert_yaxis()
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("% of products in the group")
    ax2.set_title("How additive codes are written, by market", loc="left", color=INK)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False, fontsize=8)
    ax2.grid(False)
    save(fig, "04_additive_code_frequency.png")


def fig_function_classes(class_pct):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(class_pct.index, class_pct.values, color=BLUE, height=0.65)
    ax.invert_yaxis()
    ax.set_xlabel("% of products with at least one additive of this class")
    ax.set_title("Possible functional classes of detected additives", loc="left", color=INK)
    ax.text(0, -0.13, "Source: OFF additive detection + OFF taxonomy. One additive can have several classes\n"
            "(e.g. citric acid: antioxidant, sequestrant), so bars do not sum to 100%.",
            transform=ax.transAxes, fontsize=7.5, color=MUTED, va="top")
    label_bars(ax, class_pct.values, "{:.1f}%")
    save(fig, "05_additive_function_classes.png")


def fig_groups_and_categories(df, top_n=15):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [0.7, 1.3]})
    fig.subplots_adjust(wspace=0.55)
    groups = df["country_group"].value_counts().reindex(GROUP_ORDER).fillna(0)
    ax1.barh(groups.index, groups.values, color=BLUE, height=0.6)
    ax1.invert_yaxis()
    ax1.set_xlabel("products")
    ax1.set_title("Products per market group", loc="left", color=INK)
    ax1.set_xlim(right=groups.max() * 1.2)
    label_bars(ax1, groups.values)

    # OFF uses both empty cells and the literal "Undefined" for a missing category
    cats = df["main_category"].replace("Undefined", None).fillna("(no category)").value_counts().head(top_n)
    ax2.barh(cats.index, cats.values, color=BLUE, height=0.65)
    ax2.invert_yaxis()
    ax2.set_xlabel("products")
    ax2.set_title(f"Top {top_n} main categories", loc="left", color=INK)
    label_bars(ax2, cats.values)
    save(fig, "06_market_groups_and_categories.png")


# --- Main -----------------------------------------------------------------------------------

def main():
    config = load_config()
    out = config["outputs"]
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    profile = json.loads(project_path(out["source_profile"]).read_text(encoding="utf-8"))
    filter_log = json.loads(project_path(out["filter_log"]).read_text(encoding="utf-8"))
    df = pd.read_csv(project_path(out["products"]), dtype={"product_id": str})
    additives = pd.read_csv(project_path("data/processed/additives_reference.csv"), dtype=str).fillna("")
    n = len(df)

    df["segments"] = df["ingredients_text_raw"].map(crude_segments)
    df["n_segments"] = df["segments"].map(len)
    df["code_style"] = df["ingredients_text_raw"].map(code_style)

    # Each segment counted at most once per product ("document frequency")
    segment_counts = Counter(s for segs in df["segments"] for s in set(segs))

    # Additives according to Open Food Facts' own detection ("en:e330,en:e955")
    off_tags = df["off_additives_tags"].fillna("").map(
        lambda s: sorted({t.replace("en:e", "") for t in s.split(",") if t.startswith("en:e")}))
    df["has_additive"] = off_tags.map(bool)
    tag_counts = Counter(t for tags in off_tags for t in tags)
    names = dict(zip(additives["number"], additives["name"]))
    top_additives = [(f"E{code}", names.get(code, "?"), c) for code, c in tag_counts.most_common(20)]

    classes = dict(zip(additives["number"], additives["function_classes"]))
    class_products = Counter()
    for tags in off_tags:
        product_classes = {c for t in tags for c in classes.get(t, "").split("|") if c}
        class_products.update(product_classes)
    class_pct = (pd.Series(class_products) * 100 / n).sort_values(ascending=False).head(15)
    class_pct.index = [c.replace("-", " ") for c in class_pct.index]

    style_table = (pd.crosstab(df["country_group"], df["code_style"], normalize="index") * 100).reindex(GROUP_ORDER)

    # Codes actually written in the text (any notation), normalised to the bare number
    text_codes = Counter()
    for text in df["ingredients_text_raw"]:
        found = {re.sub(r"(?i)^(e|ins)\W*(no\.?|number)?\W*", "", m.group(0)).lower().replace(" ", "")
                 for regex in (E_CODE_RE, INS_CODE_RE) for m in regex.finditer(text)}
        text_codes.update(found)

    print("Making figures...")
    fig_dataset_size(profile, filter_log)
    fig_length_distribution(df)
    fig_top_terms(segment_counts, n)
    fig_additive_codes(top_additives, style_table, n)
    fig_function_classes(class_pct)
    fig_groups_and_categories(df)

    dup_names = df.duplicated(subset=["product_name", "brands"], keep=False) & df["product_name"].notna()
    stats = {
        "source_total_products": profile["total_products"],
        "source_missing_ingredients_percent": profile["missing_percent"]["ingredients_text"],
        "source_language_estimate": profile["language_estimate_from_sample"]["counts"],
        "project_products": n,
        "products_per_group": df["country_group"].value_counts().to_dict(),
        "chars": df["n_chars"].describe().round(1).to_dict(),
        "segments": df["n_segments"].describe().round(1).to_dict(),
        "exact_duplicate_lists_removed": filter_log["funnel"][-3]["products"] - filter_log["funnel"][-2]["products"],
        "products_sharing_text_with_another_product_before_dedup":
            int((df["n_products_with_same_text"] > 1).sum()),
        "rows_with_same_name_and_brand": int(dup_names.sum()),
        "percent_with_additive_off": round(100 * df["has_additive"].mean(), 1),
        "percent_with_additive_by_group": (df.groupby("country_group")["has_additive"].mean() * 100).round(1).to_dict(),
        "percent_with_code_style_by_group": style_table.round(1).to_dict(orient="index"),
        "percent_with_percentage_sign": round(100 * df["ingredients_text_raw"].str.contains("%").mean(), 1),
        "percent_with_brackets": round(100 * df["ingredients_text_raw"].str.contains(r"[(\[]").mean(), 1),
        "percent_all_uppercase": round(100 * df["ingredients_text_raw"].map(
            lambda t: t.upper() == t and any(ch.isalpha() for ch in t)).mean(), 1),
        "percent_non_ascii": round(100 * df["ingredients_text_raw"].map(lambda t: not t.isascii()).mean(), 1),
        "top_segments": {s: c for s, c in segment_counts.most_common(30)},
        "top_additives_off": {f"{c} {nm}": k for c, nm, k in top_additives},
        "top_codes_written_in_text": dict(text_codes.most_common(20)),
        "top_function_classes_percent": class_pct.round(1).to_dict(),
        "top_main_categories": df["main_category"].replace("Undefined", None).value_counts().head(15).to_dict(),
    }
    project_path("reports/dataset_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_markdown_report(stats, filter_log, profile)
    print("Saved reports/dataset_statistics.json and reports/dataset_statistics.md")


def write_markdown_report(s, filter_log, profile):
    lines = [
        "# Dataset statistics (auto-generated by `python -m src.data.eda`)",
        "",
        "Do not edit by hand - rerun the script instead.",
        "",
        "## 1. Source and filtering",
        "",
        f"* Full Open Food Facts export: **{s['source_total_products']:,} products**; "
        f"`ingredients_text` empty for **{s['source_missing_ingredients_percent']}%** of them.",
        f"* Project dataset: **{s['project_products']:,} products** "
        f"({', '.join(f'{k}: {v:,}' for k, v in s['products_per_group'].items())}).",
        "",
        "| Filtering step | Products remaining |",
        "|---|---:|",
        f"| Full export | {profile['total_products']:,} |",
    ]
    lines += [f"| {step['step']} | {step['products']:,} |" for step in filter_log["funnel"]]
    lines += [
        "",
        "Estimated language mix of ingredient lists in the full export "
        "(2% random sample, keyword heuristic in `src/data/language.py`; only 7 languages are "
        "detected, all others count as *unknown*):",
        "",
        "| Language | Sampled lists |", "|---|---:|",
    ]
    lines += [f"| {k} | {v:,} |" for k, v in s["source_language_estimate"].items()]
    lines += [
        "",
        "![size](figures/01_dataset_size_and_missingness.png)",
        "",
        "## 2. Text length",
        "",
        f"* Characters: mean {s['chars']['mean']}, median {s['chars']['50%']}, "
        f"min {s['chars']['min']:.0f}, max {s['chars']['max']:.0f}.",
        f"* Segments (crude split): mean {s['segments']['mean']}, median {s['segments']['50%']}, "
        f"max {s['segments']['max']:.0f}.",
        "",
        "![length](figures/02_ingredient_length_distribution.png)",
        "",
        "## 3. Duplicates and text properties",
        "",
        f"* Exact-duplicate ingredient lists removed during filtering: {s['exact_duplicate_lists_removed']:,}.",
        f"* Kept products whose list was shared by >1 product before de-duplication: "
        f"{s['products_sharing_text_with_another_product_before_dedup']:,}.",
        f"* Rows sharing product name + brand with another row (near-duplicates, handled by the "
        f"grouped split later): {s['rows_with_same_name_and_brand']:,}.",
        f"* Lists containing a percentage: {s['percent_with_percentage_sign']}%; containing brackets: "
        f"{s['percent_with_brackets']}%; written ALL UPPERCASE: {s['percent_all_uppercase']}%; "
        f"containing non-ASCII characters: {s['percent_non_ascii']}%.",
        "",
        "## 4. Ingredients and additives",
        "",
        f"* Products with at least one additive (Open Food Facts detection): **{s['percent_with_additive_off']}%**; "
        f"by group: {', '.join(f'{k} {v}%' for k, v in s['percent_with_additive_by_group'].items())}.",
        "",
        "![terms](figures/03_top_ingredient_terms.png)",
        "",
        "![codes](figures/04_additive_code_frequency.png)",
        "",
        "Most frequent codes written in the text (E or INS notation, normalised to the number): "
        + ", ".join(f"{k} ({v})" for k, v in s["top_codes_written_in_text"].items()) + ".",
        "",
        "![classes](figures/05_additive_function_classes.png)",
        "",
        "![groups](figures/06_market_groups_and_categories.png)",
        "",
    ]
    project_path("reports/dataset_statistics.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
