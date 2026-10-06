"""Ingredient Scanner - Streamlit web app.

Run from the project root:
    streamlit run src/app/streamlit_app.py

Take a photo of an ingredient list (or upload one / paste text). The app reads the text (RapidOCR +
food-dictionary spelling correction), finds the ingredient section, classifies every ingredient (hybrid:
dictionary rules + fine-tuned DistilBERT + fuzzy knowledge-base matching), links it to the knowledge base,
explains it, lists allergens, and compares a portion with daily reference intakes (WHO / EU / EFSA / JECFA).
"""
import hashlib
import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.app.intake import PROFILES, additive_guidance, assess_portion, headline, special_notes
from src.app.nutrition import NUTRIENTS, estimate_from_similar_product
from src.app.scanner import CATEGORY_ORDER, CATEGORY_TITLES, DISCLAIMER, FINAL_MODEL_DIR, IngredientScanner

COLOURS = {"SUGAR": "#eb6834", "SWEETENER": "#e87ba4", "FAT": "#c98500", "PRESERVATIVE": "#e34948",
           "COLOUR": "#4a3aa7", "ADDITIVE": "#2a78d6", "INS_CODE": "#1baf7a", "FUNCTION_CLASS": "#52514e",
           "FLAVOURING": "#008300", "INGREDIENT": "#898781"}
LABEL_NAMES = {"SUGAR": "Sugar", "SWEETENER": "Sweetener", "FAT": "Fat / oil", "PRESERVATIVE": "Preservative",
               "COLOUR": "Colour", "ADDITIVE": "Additive", "INS_CODE": "INS / E code",
               "FUNCTION_CLASS": "Function stated on label", "FLAVOURING": "Flavouring", "INGREDIENT": "Ingredient"}
EXAMPLE_TEXT = ("Ingredients: Refined wheat flour (maida), sugar, refined palm oil, invert syrup, dextrose, "
                "raising agents [503(ii), 500(ii)], iodised salt, emulsifier (INS 322), acidity regulator (INS 330), "
                "colour (150d), preservative (211), artificial flavouring substances (vanilla).")
LIGHT_ICON = {"low": "🟢 low", "medium": "🟠 medium", "high": "🔴 high", "": ""}
EDITABLE_NUTRIENTS = ["energy_kcal", "fat_g", "saturated_fat_g", "carbohydrate_g", "sugars_g", "fibre_g",
                      "protein_g", "salt_g"]
CLASSIFIERS = {"Hybrid (recommended)": "hybrid", "DistilBERT only": "transformer", "Dictionary rules only": "dictionary"}

st.set_page_config(page_title="Ingredient Scanner", page_icon="🔍", layout="wide")


# ------------------------------------------------------------------------------------ cached work
@st.cache_resource(show_spinner="Loading the classifier ...")
def get_scanner(kind: str) -> IngredientScanner:
    return IngredientScanner(kind)


def prepare_photo(image_bytes: bytes, rotation: int = 0, crop: tuple = (0, 100, 0, 100)):
    """Rotate and crop (left, right, top, bottom in % of the photo) -> BGR array."""
    import cv2
    from src.ocr.ocr_pipeline import load_image
    image = load_image(image_bytes)
    if rotation:
        codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
        image = cv2.rotate(image, codes[rotation])
    left, right, top, bottom = crop
    h, w = image.shape[:2]
    if (left, right, top, bottom) != (0, 100, 0, 100) and right > left and bottom > top:
        image = image[int(h * top / 100):int(h * bottom / 100), int(w * left / 100):int(w * right / 100)]
    return image


@st.cache_data(show_spinner=False, max_entries=30)
def scan_image_cached(image_bytes: bytes, kind: str, rotation: int = 0, crop: tuple = (0, 100, 0, 100)) -> dict:
    return get_scanner(kind).scan_image(prepare_photo(image_bytes, rotation, crop))


@st.cache_data(show_spinner=False, max_entries=100)
def scan_text_cached(text: str, kind: str) -> dict:
    return get_scanner(kind).scan_text(text)


@st.cache_data(show_spinner=False, max_entries=100)
def estimate_cached(text: str) -> dict:
    return estimate_from_similar_product(text)


@st.cache_data
def example_photos() -> pd.DataFrame:
    index = ROOT / "data/images/packets.csv"
    products = ROOT / "data/processed/products.csv"
    if not index.exists():
        return pd.DataFrame()
    photos = pd.read_csv(index, dtype={"product_id": str})
    names = pd.read_csv(products, dtype={"product_id": str}, usecols=["product_id", "product_name", "brands"])
    photos = photos.merge(names, on="product_id", how="left")
    photos["title"] = (photos["product_name"].fillna("Unnamed product") + " - " + photos["brands"].fillna("")
                       + " (" + photos["country_group"] + ")")
    return photos


# ------------------------------------------------------------------------------------ rendering
def highlighted(text: str, entities: list) -> str:
    parts, pos = [], 0
    for e in entities:
        if e["start"] < pos:
            continue
        parts.append(html.escape(text[pos:e["start"]]))
        title = html.escape(f"{LABEL_NAMES[e['label']]}" + (f" - {e['canonical_name']}" if e["canonical_name"] else "")
                            + f" (found by: {e.get('source', '')})")
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


def show_classification(result: dict):
    summary = result["summary"]
    st.subheader("1 · What is in it")
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
    if hidden.get("sugar_like_carbohydrates"):
        notes.append("**Sugar-like carbohydrates:** " + ", ".join(hidden["sugar_like_carbohydrates"])
                     + " - not counted as 'sugars' on the label, but digested quickly into glucose.")
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
        with (left if i % 2 == 0 else right).expander(f"{CATEGORY_TITLES[cat]} ({len(items)})",
                                                       expanded=bool(items) and cat != "INGREDIENT"):
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


def show_allergens(result: dict):
    allergens = result.get("allergens") or {}
    contains, may = allergens.get("contains", {}), allergens.get("may_contain", {})
    st.subheader("2 · Allergens")
    if not contains and not may:
        st.success("None of the 14 major allergen groups was found in the text. (Check the label: OCR can miss words.)")
        return
    if contains:
        st.error("**Contains:** " + " · ".join(f"{g} ({', '.join(w)})" for g, w in contains.items()))
    if may:
        st.warning("**May contain (precautionary):** " + " · ".join(f"{g} ({', '.join(w)})" for g, w in may.items()))


def nutrition_inputs(result: dict, rid: str) -> tuple:
    """Per-100 g values: from the photo, else an estimate from a similar product; always editable."""
    parsed = result.get("nutrition") or {}
    values, source, serving = dict(parsed.get("per_100g") or {}), parsed.get("source", ""), parsed.get("serving_g")
    if parsed.get("basis", "").startswith("converted"):
        source += f" ({parsed['basis']})"
    key_nutrients = ["sugars_g", "fat_g", "saturated_fat_g", "salt_g"]
    if len(values) < 3 or any(k not in values for k in key_nutrients):
        estimate = estimate_cached(result["input_text"])
        if estimate:
            reference = (f"the most similar product in our Open Food Facts database: *{estimate['product_name']}* "
                         f"(ingredient-list similarity {estimate['similarity']:.0%})")
            if len(values) < 3:
                values = dict(estimate["per_100g"])
                source = "**estimated** from " + reference
            else:
                filled = [NUTRIENTS[k][0].lower() for k in key_nutrients if k not in values and k in estimate["per_100g"]]
                for k in key_nutrients:
                    values.setdefault(k, estimate["per_100g"].get(k))
                if filled:
                    source += f"; **{', '.join(filled)} estimated** from " + reference
    if not source:
        source = "no nutrition table found - please type the values from the packet"
    st.caption("Nutrition per 100 g / 100 ml - source: " + source + ". You can correct any number.")
    cols = st.columns(4)
    edited = {}
    for i, key in enumerate(EDITABLE_NUTRIENTS):
        name, unit = NUTRIENTS[key]
        default = values.get(key)
        edited[key] = cols[i % 4].number_input(f"{name} ({unit})", min_value=0.0,
                                               max_value=900.0 if unit == "kcal" else 100.0,
                                               value=float(default) if default is not None else None, step=0.1,
                                               format="%.1f", key=f"{rid}-{key}", placeholder="not given")
    return {k: v for k, v in edited.items() if v is not None}, serving


def show_daily_guide(result: dict, rid: str):
    st.subheader("3 · Is this amount OK? Daily intake guide")
    st.caption("Compares a portion of this product with official daily reference values (WHO, EU, EFSA). "
               "General guidance for healthy people - not personal medical advice.")
    per100, serving = nutrition_inputs(result, rid)

    c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
    profile_name = c1.selectbox("Who is eating?", list(PROFILES), key=f"{rid}-profile")
    profile = PROFILES[profile_name]
    # a drink: typical drink words, or a nutrition table given per 100 ml ("water" alone is not enough: sauces)
    full = result.get("ocr", {}).get("full_text", result["input_text"])
    drinkish = bool(re.search(r"carbonated|beverage|soft drink|energy drink|\bcola\b|juice drink|nectar", result["text"], re.I)
                    or re.search(r"(?:per|/)\s*100\s*ml", full, re.I))
    kind = c2.radio("Product type", ["food", "drink"], index=1 if drinkish else 0, horizontal=True, key=f"{rid}-kind")
    unit = "ml" if kind == "drink" else "g"
    portion = c3.number_input(f"Portion you eat ({unit})", min_value=1.0, max_value=2000.0,
                              value=float(serving) if serving else (250.0 if kind == "drink" else 30.0),
                              step=5.0, key=f"{rid}-portion")
    weight = c4.number_input("Body weight (kg)", min_value=5.0, max_value=200.0, value=float(profile.body_weight_kg),
                             step=1.0, key=f"{rid}-weight-{profile_name}")

    if per100:
        rows = assess_portion(per100, portion, profile, kind)
        text, level = headline(rows, portion)
        {"ok": st.success, "caution": st.warning, "high": st.error}.get(level, st.info)("**" + text + "**")

        for r in [r for r in rows if r["limit_kind"] == "max"]:
            st.markdown(f"**{r['nutrient']}** - {r['in_portion']:g} {r['unit']} in your portion · daily limit "
                        f"{r['daily_value']:g} {r['unit']} · {r['message']}"
                        + (f" · per 100 {unit}: {LIGHT_ICON[r['traffic_light']]}" if r["traffic_light"] else ""))
            st.progress(min(r["share"], 1.0), text=f"{r['share']:.0%} of the daily limit")
            if r["max_product_g"]:
                st.caption(f"↳ {r['max_product_g']:,} {unit} of this product alone would reach the whole day's limit "
                           f"for {r['nutrient'].lower()}. Source: {r['source']}.")
        other = [r for r in rows if r["limit_kind"] != "max"]
        if other:
            st.markdown("**Other nutrients in your portion**")
            st.dataframe(pd.DataFrame([{"Nutrient": r["nutrient"], "In portion": f"{r['in_portion']:g} {r['unit']}",
                                        "Daily reference": f"{r['daily_value']:g} {r['unit']}",
                                        "Share": r["share"], "Comment": r["message"]} for r in other]),
                         hide_index=True, width="stretch",
                         column_config={"Share": st.column_config.ProgressColumn(format="percent", min_value=0,
                                                                                 max_value=1)})
    else:
        st.info("Type the nutrition values from the packet above to see how a portion compares with daily limits.")

    st.markdown("##### Additives: acceptable daily intake (ADI)")
    st.caption("ADI = the amount that can be eaten every day over a lifetime without appreciable health risk "
               "(JECFA / EFSA). Labels rarely state how much additive is inside; permitted use levels are set so "
               "that normal portions stay well below the ADI.")
    adi_rows = additive_guidance(result["entities"], weight)
    if adi_rows:
        st.dataframe(pd.DataFrame([{"Additive": r["additive"], "Daily limit (ADI)": r["daily_limit"],
                                    "Evaluated by": r["authority"], "Note": r["note"]} for r in adi_rows]),
                     hide_index=True, width="stretch")
    else:
        st.write("No additive with a known INS number was found.")
    for note in special_notes(result["text"] + " " + result.get("ocr", {}).get("full_text", "")):
        st.warning(f"**{note['substance']}:** {note['note']}")

    with st.expander("Where do these numbers come from?"):
        st.markdown(
            "- **Sugars:** WHO guideline *Sugars intake for adults and children* (2015): free sugars below 10 % of "
            "energy (≈ 50 g at 2000 kcal), ideally below 5 % (≈ 25 g). ICMR-NIN *Dietary Guidelines for Indians* "
            "(2024) also advise keeping added sugar low. The label's 'sugars' include natural milk/fruit sugars, so "
            "the comparison is an upper estimate.\n"
            "- **Salt:** WHO (2012): less than 5 g salt (2 g sodium) per day for adults, less for young children.\n"
            "- **Fats:** WHO (2023): saturated fat below 10 % and trans fat below 1 % of energy; total fat below 30 %.\n"
            "- **Reference intakes:** EU Regulation 1169/2011, Annex XIII (energy 2000 kcal, fat 70 g, saturates "
            "20 g, carbohydrate 260 g, sugars 90 g, protein 50 g, salt 6 g). Fibre: EFSA (2010), 25 g/day.\n"
            "- **Traffic lights:** UK Food Standards Agency front-of-pack criteria per 100 g / 100 ml.\n"
            "- **Additives:** acceptable daily intakes from JECFA (FAO/WHO) and EFSA scientific opinions "
            "(`data/knowledge_base/additive_adi.csv`).")


def show_details(result: dict):
    with st.expander("Details: every entity, how it was found and linked"):
        st.dataframe(pd.DataFrame(result["entities"])[["text", "label", "source", "canonical_name", "ins_number",
                                                        "function", "link_method", "link_score"]],
                     hide_index=True, width="stretch")
    if "ocr" in result:
        o = result["ocr"]
        st.caption(f"OCR: {o['engine']} · enlarged ×{o['scale']} · rotation {o['rotation']}° · {o['n_lines']} lines · "
                   f"mean confidence {o['mean_confidence']:.2f} · ingredient section found by: {o['section_method']} · "
                   f"{o['seconds']} s")


# ------------------------------------------------------------------------------------ page
st.title("🔍 Ingredient Scanner")
st.caption("Photograph an ingredient list. The app reads it, finds sugars, fats, sweeteners, preservatives, colours, "
           "additives and INS/E codes, lists allergens, and shows how a portion compares with daily limits.")

with st.sidebar:
    st.header("Settings")
    has_model = (FINAL_MODEL_DIR / "config.json").exists()
    options = [name for name, k in CLASSIFIERS.items() if k != "transformer" or has_model]
    choice = st.radio("Classifier", options,
                      help="Hybrid = dictionary rules + fine-tuned DistilBERT + fuzzy matching against the knowledge base.")
    kind = CLASSIFIERS[choice]
    st.markdown("**Pipeline**\n\n1. OCR (RapidOCR PP-OCRv6), auto-enlarging small print\n2. Spelling correction "
                "(food dictionary)\n3. Find the ingredients section\n4. Named-entity recognition\n5. Link to knowledge "
                "base\n6. Allergens + daily intake guide")
    st.markdown("**Tips for photos**\n\n- Fill the frame with the ingredient list (and the nutrition table)\n"
                "- Good light, no glare, hold the camera steady\n- Curved packs: photograph the flat part")
    st.caption(DISCLAIMER)

if "active" not in st.session_state:
    st.session_state.active = None          # ("image", bytes, rotation, crop) or ("text", str)

tab_upload, tab_camera, tab_text, tab_example = st.tabs(["🖼️ Upload photo", "📷 Camera", "⌨️ Paste text",
                                                         "📦 Example packets"])

with tab_upload:
    upload = st.file_uploader("Photo of an ingredient list (JPG / PNG / WEBP)", type=["jpg", "jpeg", "png", "webp"])
    if upload is not None:
        data = upload.getvalue()
        with st.expander("Rotate / crop (optional - helps when the list is a small part of the photo)"):
            rot_u = st.radio("Rotate", [0, 90, 180, 270], horizontal=True, format_func=lambda a: f"{a}°",
                             key="rot_upload")
            c_left, c_right = st.columns(2)
            x_range = c_left.slider("Keep from left to right (%)", 0, 100, (0, 100), key="crop_x")
            y_range = c_right.slider("Keep from top to bottom (%)", 0, 100, (0, 100), key="crop_y")
        crop = (x_range[0], x_range[1], y_range[0], y_range[1])
        try:
            import cv2
            preview = cv2.cvtColor(prepare_photo(data, rot_u, crop), cv2.COLOR_BGR2RGB)
            st.image(preview, width=360, caption="Photo that will be read")
        except Exception:
            st.error("This file could not be opened as an image.")
        signature = hashlib.md5(data).hexdigest()
        if st.session_state.get("upload_sig") != signature:          # a new photo -> scan it automatically
            st.session_state.upload_sig = signature
            st.session_state.active = ("image", data, rot_u, crop)
        if st.button("Scan this photo (after rotating / cropping)", type="primary", key="scan_upload"):
            st.session_state.active = ("image", data, rot_u, crop)

with tab_camera:
    shot = st.camera_input("Point the camera at the ingredient list")
    if shot is not None:
        data = shot.getvalue()
        signature = hashlib.md5(data).hexdigest()
        if st.session_state.get("camera_sig") != signature:
            st.session_state.camera_sig = signature
            st.session_state.active = ("image", data, 0, (0, 100, 0, 100))

with tab_text:
    typed = st.text_area("Ingredient list (you can also paste the nutrition table)", EXAMPLE_TEXT, height=140)
    if st.button("Analyse text", type="primary", key="scan_text"):
        st.session_state.active = ("text", typed)

with tab_example:
    photos = example_photos()
    if photos.empty:
        st.write("No example photos found (run `python -m src.ocr.collect_images`).")
    else:
        title = st.selectbox("Real packet photos (Open Food Facts, CC BY-SA)", photos["title"].tolist())
        row = photos[photos["title"] == title].iloc[0]
        st.image(str(ROOT / row["image"]), width=320)
        if st.button("Scan this packet", type="primary", key="scan_example"):
            st.session_state.active = ("image", (ROOT / row["image"]).read_bytes(), 0, (0, 100, 0, 100))

active = st.session_state.active
result = None
if active is not None:
    if active[0] == "image":
        with st.spinner("Reading the label (OCR on CPU: usually 3-15 seconds) ..."):
            try:
                result = scan_image_cached(active[1], kind, active[2], active[3])
            except Exception as err:
                st.error(f"This file could not be read as an image: {err}")
        if result is not None:
            o = result["ocr"]
            if o["mean_confidence"] < 0.8 or len(result["entities"]) < 3:
                st.warning("This photo is hard to read (blurred, curved, glare or very small print), so some "
                           "ingredients may be missing. Try: crop to the ingredient list (Upload tab → Rotate / "
                           "crop), a sharper photo in good light, or correct the text below and press Re-analyse.")
            st.divider()
            st.markdown("#### Text read from the photo")
            st.caption("Check it against the packet. If a word was misread, correct it here and press *Re-analyse*.")
            ocr_text = st.text_area("OCR text", result["ocr"]["full_text"], height=170, label_visibility="collapsed",
                                    key="ocr_" + hashlib.md5(result["ocr"]["full_text"].encode()).hexdigest())
            if st.button("Re-analyse corrected text", key="reanalyse"):
                st.session_state.active = ("text", ocr_text)
                st.rerun()
    else:
        st.divider()
        st.caption("Showing the result for the text from the *Paste text* tab (or your corrected OCR text).")
        result = scan_text_cached(active[1], kind)

if result is not None:
    rid = hashlib.md5((result["input_text"] + kind).encode()).hexdigest()[:10]
    if not result["entities"]:
        st.warning("No ingredients were recognised. Try a sharper, well-lit photo that shows the ingredient list, "
                   "rotate it, or paste the text in the *Paste text* tab.")
    else:
        st.caption(f"Classifier: {result['ner_model']} · ingredient section: "
                   f"{result.get('ocr', {}).get('section_method', result.get('section_method', ''))}")
        show_classification(result)
        st.divider()
        show_allergens(result)
        st.divider()
        show_daily_guide(result, rid)
        st.divider()
        show_details(result)
    st.caption(DISCLAIMER)
