"""Ingredient Scanner - Streamlit web app.

Run from the project root:
    streamlit run src/app/streamlit_app.py

Take a photo of an ingredient list (or upload one / paste text). The app reads the text (EasyOCR),
finds the ingredient section, classifies every ingredient (fine-tuned DistilBERT, or dictionary
rules), links it to the knowledge base and explains it in plain language.
"""
import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.app.scanner import CATEGORY_ORDER, CATEGORY_TITLES, DISCLAIMER, FINAL_MODEL_DIR, IngredientScanner

COLOURS = {"SUGAR": "#eb6834", "SWEETENER": "#e87ba4", "FAT": "#c98500", "PRESERVATIVE": "#e34948",
           "COLOUR": "#4a3aa7", "ADDITIVE": "#2a78d6", "INS_CODE": "#1baf7a", "FUNCTION_CLASS": "#52514e",
           "FLAVOURING": "#008300", "INGREDIENT": "#898781"}
LABEL_NAMES = {"SUGAR": "Sugar", "SWEETENER": "Sweetener", "FAT": "Fat / oil", "PRESERVATIVE": "Preservative",
               "COLOUR": "Colour", "ADDITIVE": "Additive", "INS_CODE": "INS / E code",
               "FUNCTION_CLASS": "Function stated on label", "FLAVOURING": "Flavouring", "INGREDIENT": "Ingredient"}
EXAMPLE_TEXT = ("Refined wheat flour (maida), sugar, refined palm oil, invert syrup, dextrose, raising agents "
                "[503(ii), 500(ii)], iodised salt, emulsifier (INS 322), acidity regulator (INS 330), "
                "colour (150d), preservative (211), artificial flavouring substances (vanilla).")

st.set_page_config(page_title="Ingredient Scanner", page_icon="🔍", layout="wide")


@st.cache_resource(show_spinner="Loading the NER model ...")
def get_scanner(kind: str) -> IngredientScanner:
    return IngredientScanner(kind)


@st.cache_data
def example_photos() -> pd.DataFrame:
    index = ROOT / "data/images/packets.csv"
    products = ROOT / "data/processed/products.csv"
    if not index.exists():
        return pd.DataFrame()
    photos = pd.read_csv(index, dtype={"product_id": str})
    names = pd.read_csv(products, dtype={"product_id": str}, usecols=["product_id", "product_name", "brands"])
    photos = photos.merge(names, on="product_id", how="left")
    quality = ROOT / "reports/ocr_per_photo.csv"          # measured OCR readability of each photo
    if quality.exists():
        q = pd.read_csv(quality, dtype={"product_id": str})[["product_id", "word_recall_full_ocr"]]
        photos = photos.merge(q, on="product_id", how="left").sort_values("word_recall_full_ocr", ascending=False)
        photos["quality"] = photos["word_recall_full_ocr"].map(lambda r: "clear photo" if r >= 0.8 else
                                                                 "readable" if r >= 0.5 else "hard photo")
    else:
        photos["quality"] = "photo"
    photos["title"] = (photos["product_name"].fillna("Unnamed product") + " - " + photos["brands"].fillna("")
                       + " (" + photos["country_group"] + ", " + photos["quality"] + ")")
    return photos


def highlighted(text: str, entities: list) -> str:
    parts, pos = [], 0
    for e in entities:
        parts.append(html.escape(text[pos:e["start"]]))
        title = html.escape(f"{LABEL_NAMES[e['label']]}" + (f" - {e['canonical_name']}" if e["canonical_name"] else ""))
        parts.append(f'<span title="{title}" style="background:{COLOURS[e["label"]]};color:white;'
                     f'padding:1px 5px;border-radius:5px;white-space:nowrap">{html.escape(text[e["start"]:e["end"]])}'
                     f'<sub style="font-size:0.62em;opacity:0.9"> {e["label"].replace("_", " ").lower()}</sub></span>')
        pos = e["end"]
    parts.append(html.escape(text[pos:]))
    return '<div style="line-height:2.3;font-size:1.02rem">' + "".join(parts) + "</div>"


def legend() -> str:
    chips = "".join(f'<span style="background:{c};color:white;padding:1px 7px;border-radius:5px;margin:0 6px 6px 0;'
                    f'display:inline-block;font-size:0.8rem">{LABEL_NAMES[l]}</span>' for l, c in COLOURS.items())
    return f"<div>{chips}</div>"


def show_result(result: dict):
    summary = result["summary"]
    st.subheader("What we found")
    cols = st.columns(len(CATEGORY_ORDER))
    for col, cat in zip(cols, CATEGORY_ORDER):
        col.metric(CATEGORY_TITLES[cat], summary["counts"][cat])

    st.markdown("##### Ingredient list, classified")
    st.markdown(legend(), unsafe_allow_html=True)
    st.markdown(highlighted(result["text"], result["entities"]), unsafe_allow_html=True)

    hidden = summary["hidden"]
    notes = []
    if hidden["sugars_under_other_names"]:
        notes.append("**Sugars listed under other names:** " + ", ".join(hidden["sugars_under_other_names"]))
    if hidden["fats_under_other_names"]:
        notes.append("**Fats listed under other names:** " + ", ".join(hidden["fats_under_other_names"]))
    codes = [e for e in result["entities"] if e["label"] == "INS_CODE"]
    if codes:
        notes.append("**Additives written as codes:** " + ", ".join(
            f"{e['text']} = {e['canonical_name'] or 'unknown code'}" for e in codes))
    if notes:
        st.info("\n\n".join(notes))

    st.markdown("##### By category")
    left, right = st.columns(2)
    for i, cat in enumerate(CATEGORY_ORDER):
        items = summary["groups"][cat]
        with (left if i % 2 == 0 else right).expander(f"{CATEGORY_TITLES[cat]} ({len(items)})", expanded=bool(items) and cat != "INGREDIENT"):
            st.write(", ".join(items) if items else "None found.")

    additives = [e for e in result["entities"] if e["label"] in ("INS_CODE", "ADDITIVE", "PRESERVATIVE", "COLOUR", "SWEETENER")]
    if additives:
        st.markdown("##### Additives explained")
        table = pd.DataFrame([{
            "On the label": e["text"],
            "Identified as": e["canonical_name"] or "-",
            "INS / E": f"INS {e['ins_number']}" if e["ins_number"] else "-",
            "Used as": (e["function"] or "-") + (" (stated on label)" if e["function_source"] == "label" else ""),
            "What that means": e["description"],
            "More information": e["more_info_url"] or None,
        } for e in additives])
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={"More information": st.column_config.LinkColumn(display_text="Wikidata")})

    with st.expander("Details: every entity and how it was linked"):
        st.dataframe(pd.DataFrame(result["entities"])[["text", "label", "canonical_name", "ins_number", "function",
                                                        "link_method", "link_score"]],
                     hide_index=True, width="stretch")
    if "ocr" in result:
        with st.expander("Details: OCR"):
            o = result["ocr"]
            st.caption(f"Section found by: {o['section_method']} · rotation {o['rotation']}° · "
                       f"mean OCR confidence {o['mean_confidence']:.2f} · {o['seconds']} s")
            st.text_area("Full OCR text", o["full_text"], height=180)


# ------------------------------------------------------------------------------------ page
st.title("🔍 Ingredient Scanner")
st.caption("Take a picture of an ingredient list. The app finds sugars, fats, additives, INS/E codes and other "
           "ingredients, and explains what they are.")

with st.sidebar:
    st.header("Settings")
    has_model = (FINAL_MODEL_DIR / "config.json").exists()
    options = (["DistilBERT (fine-tuned)"] if has_model else []) + ["Dictionary rules"]
    choice = st.radio("Classifier", options, help="DistilBERT is more robust to OCR errors; the rules are the baseline.")
    kind = "transformer" if choice.startswith("DistilBERT") else "dictionary"
    st.markdown("**Pipeline**\n\n1. OCR (EasyOCR)\n2. Find the ingredients section\n3. Normalise text\n"
                "4. Named-entity recognition\n5. Link to knowledge base\n6. Explain")
    st.caption(DISCLAIMER)

scanner = get_scanner(kind)
tab_camera, tab_upload, tab_example, tab_text = st.tabs(["📷 Camera", "🖼️ Upload photo", "📦 Example packets", "⌨️ Paste text"])
result, image_bytes = None, None

with tab_camera:
    shot = st.camera_input("Point the camera at the ingredient list")
    if shot is not None:
        image_bytes = shot.getvalue()
with tab_upload:
    upload = st.file_uploader("Photo of an ingredient list", type=["jpg", "jpeg", "png", "webp"])
    if upload is not None:
        image_bytes = upload.getvalue()
with tab_example:
    photos = example_photos()
    if photos.empty:
        st.write("No example photos found (run `python -m src.ocr.collect_images`).")
    else:
        title = st.selectbox("Real packet photos (Open Food Facts, CC BY-SA)", photos["title"].tolist())
        row = photos[photos["title"] == title].iloc[0]
        st.image(str(ROOT / row["image"]), width=360)
        if st.button("Scan this packet", type="primary"):
            image_bytes = (ROOT / row["image"]).read_bytes()
with tab_text:
    text = st.text_area("Ingredient list", EXAMPLE_TEXT, height=120)
    if st.button("Analyse text", type="primary"):
        result = scanner.scan_text(text)

if image_bytes is not None:
    with st.spinner("Reading the label (OCR runs on CPU, this can take 10-60 seconds) ..."):
        result = scanner.scan_image(image_bytes)

if result:
    if not result["entities"]:
        st.warning("No ingredients were recognised. Try a sharper, well-lit photo of the ingredient list only.")
        if "ocr" in result:
            with st.expander("Details: OCR", expanded=True):
                st.text_area("Text read from the photo", result["ocr"]["full_text"], height=180)
    else:
        st.caption(f"Classifier: {result['ner_model']}")
        show_result(result)
    st.divider()
    st.caption(DISCLAIMER)
