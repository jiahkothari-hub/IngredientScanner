"""Ingredient Scanner - Streamlit web app.

Run from the project root:
    streamlit run src/app/streamlit_app.py

Take a photo of an ingredient list (or upload one / paste text). The app reads the text (RapidOCR +
food-dictionary spelling correction), finds the ingredient section, classifies every ingredient (hybrid:
dictionary rules + fine-tuned DistilBERT + fuzzy knowledge-base matching), links it to the knowledge base,
explains it, lists allergens, and compares a portion with daily reference intakes (WHO / EU / EFSA / JECFA).

The look (theme CSS and HTML components) lives in src/app/ui_components.py; this file is the page logic.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.app import ui_components as ui
from src.app.intake import PROFILES, additive_guidance, assess_portion, headline, special_notes
from src.app.nutrition import NUTRIENTS, estimate_from_similar_product
from src.app.scanner import CATEGORY_ORDER, DISCLAIMER, FINAL_MODEL_DIR, IngredientScanner

EXAMPLE_TEXT = ("Ingredients: Refined wheat flour (maida), sugar, refined palm oil, invert syrup, dextrose, "
                "raising agents [503(ii), 500(ii)], iodised salt, emulsifier (INS 322), acidity regulator (INS 330), "
                "colour (150d), preservative (211), artificial flavouring substances (vanilla).")
EDITABLE_NUTRIENTS = ["energy_kcal", "fat_g", "saturated_fat_g", "carbohydrate_g", "sugars_g", "fibre_g",
                      "protein_g", "salt_g"]
KEY_NUTRIENTS = ["sugars_g", "fat_g", "saturated_fat_g", "salt_g"]
CLASSIFIERS = {"Hybrid (recommended)": "hybrid", "DistilBERT only": "transformer", "Dictionary rules only": "dictionary"}
ADDITIVE_LABELS = ("INS_CODE", "ADDITIVE", "PRESERVATIVE", "COLOUR", "SWEETENER")
SAMPLES = ["Nutella", "Diet Coke", "Maggi Masala", "Pringles Hot&Spicy", "Haribo Goldbears", "Parle-G"]
NO_CROP = (0, 100, 0, 100)

st.set_page_config(page_title="Ingredient Scanner", page_icon="🥗", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------------------------ cached work
@st.cache_resource(show_spinner="Loading the classifier ...")
def get_scanner(kind: str) -> IngredientScanner:
    return IngredientScanner(kind)


def prepare_photo(image_bytes: bytes, rotation: int = 0, crop: tuple = NO_CROP):
    """Rotate and crop (left, right, top, bottom in % of the photo) -> BGR array."""
    import cv2
    from src.ocr.ocr_pipeline import load_image
    image = load_image(image_bytes)
    if rotation:
        codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
        image = cv2.rotate(image, codes[rotation])
    left, right, top, bottom = crop
    h, w = image.shape[:2]
    if (left, right, top, bottom) != NO_CROP and right > left and bottom > top:
        image = image[int(h * top / 100):int(h * bottom / 100), int(w * left / 100):int(w * right / 100)]
    return image


@st.cache_data(show_spinner=False, max_entries=30)
def scan_image_cached(image_bytes: bytes, kind: str, rotation: int = 0, crop: tuple = NO_CROP) -> dict:
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


@st.cache_data
def sample_lists() -> dict:
    path = ROOT / "data/test_samples/typed_products.json"
    if not path.exists():
        return {}
    return {p["product"]: p["ingredients_text"] for p in json.loads(path.read_text(encoding="utf-8"))["products"]}


def html(markup: str):
    st.markdown(markup, unsafe_allow_html=True)


# ------------------------------------------------------------------------------------ nutrition state
def nutrition_defaults(result: dict) -> tuple:
    """(per-100 g values, source text, serving size): from the photo, else estimated from the most similar
    product in the Open Food Facts database (gaps in the key nutrients are filled the same way)."""
    parsed = result.get("nutrition") or {}
    values, source, serving = dict(parsed.get("per_100g") or {}), parsed.get("source", ""), parsed.get("serving_g")
    if parsed.get("basis", "").startswith("converted"):
        source += f" ({parsed['basis']})"
    if len(values) < 3 or any(k not in values for k in KEY_NUTRIENTS):
        estimate = estimate_cached(result["input_text"])
        if estimate:
            reference = (f"the most similar product in our database, *{estimate['product_name']}* "
                         f"({estimate['similarity']:.0%} ingredient match)")
            if len(values) < 3:
                values, source = dict(estimate["per_100g"]), "**estimated** from " + reference
            else:
                filled = [NUTRIENTS[k][0].lower() for k in KEY_NUTRIENTS if k not in values and k in estimate["per_100g"]]
                for k in KEY_NUTRIENTS:
                    values.setdefault(k, estimate["per_100g"].get(k))
                if filled:
                    source += f"; **{', '.join(filled)} estimated** from " + reference
    return values, source or "no nutrition table found - type the values from the packet", serving


def is_drink(result: dict) -> bool:
    full = result.get("ocr", {}).get("full_text", result["input_text"])
    return bool(re.search(r"carbonated|beverage|soft drink|energy drink|\bcola\b|juice drink|nectar", result["text"], re.I)
                or re.search(r"(?:per|/)\s*100\s*ml", full, re.I))


def current_guide(result: dict, rid: str, profile) -> dict:
    """Nutrition, portion and assessment as the user currently has them (values typed into the widgets are
    read from session state, so the summary at the top matches the guide tab)."""
    values, source, serving = nutrition_defaults(result)
    per100 = {}
    for key in EDITABLE_NUTRIENTS:
        value = st.session_state.get(f"{rid}-{key}", values.get(key))
        if value is not None:
            per100[key] = float(value)
    drink = st.session_state.get(f"{rid}-kind", "drink" if is_drink(result) else "food") == "drink"
    default_portion = float(serving) if serving else (250.0 if drink else 30.0)
    portion = float(st.session_state.get(f"{rid}-portion", default_portion))
    rows = assess_portion(per100, portion, profile, "drink" if drink else "food") if per100 else []
    return {"values": values, "source": source, "serving": serving, "per100": per100, "drink": drink,
            "portion": portion, "default_portion": default_portion, "rows": rows}


# ------------------------------------------------------------------------------------ result views
def view_summary(result: dict, guide: dict):
    summary, ents = result["summary"], result["entities"]
    sugars = summary["groups"]["SUGAR"]
    hidden_sugars = summary["hidden"]["sugars_under_other_names"] + summary["hidden"].get("sugar_like_carbohydrates", [])
    additives = sum(summary["counts"][c] for c in ("PRESERVATIVE", "COLOUR", "ADDITIVE", "SWEETENER"))
    codes = sum(1 for x in ents if x["label"] == "INS_CODE")
    contains = result.get("allergens", {}).get("contains", {})
    limited = [r for r in guide["rows"] if r["limit_kind"] == "max"]
    tiles = [
        ("Sugars", len(sugars), f"{len(hidden_sugars)} under other names" if hidden_sugars else "all named as sugar",
         ui.CATEGORY_STYLE["SUGAR"][0]),
        ("Additives", additives, f"{codes} written as codes" if codes else "no codes", ui.CATEGORY_STYLE["ADDITIVE"][0]),
        ("Allergens", len(contains), ", ".join(contains)[:60] if contains else "none of the 14 groups found",
         "#E03131" if contains else "#2F9E44"),
    ]
    if limited:
        worst = max(limited, key=lambda r: r["share"])
        unit = "ml" if guide["drink"] else "g"
        tiles.append((f"Daily limit · {guide['portion']:g} {unit}", f"{worst['share']:.0%}",
                      f"of the day's {worst['nutrient'].lower()} limit",
                      ui.LEVEL_STYLE[ui.share_level(worst["share"])][0]))
    else:
        tiles.append(("Daily limit", "-", "add nutrition values in the guide", "#5C677D"))
    html(ui.glance(tiles))


def view_ingredients(result: dict):
    summary = result["summary"]
    html(ui.category_cards(summary["groups"], CATEGORY_ORDER))

    hidden, notes = summary["hidden"], []
    if hidden["sugars_under_other_names"]:
        notes.append(("warn", "🍬 <b>Sugars under other names:</b> " + ", ".join(map(ui.e, hidden["sugars_under_other_names"]))))
    if hidden.get("sugar_like_carbohydrates"):
        notes.append(("warn", "🌾 <b>Sugar-like carbohydrates:</b> " + ", ".join(map(ui.e, hidden["sugar_like_carbohydrates"]))
                      + " - not counted as 'sugars' on the label, but digested quickly into glucose."))
    if hidden["fats_under_other_names"]:
        notes.append(("warn", "🧈 <b>Fats under other names:</b> " + ", ".join(map(ui.e, hidden["fats_under_other_names"]))))
    codes = [x for x in result["entities"] if x["label"] == "INS_CODE"]
    if codes:
        notes.append(("info", "🔢 <b>Codes decoded:</b> " + " · ".join(
            f"{ui.e(x['text'])} = {ui.e(x['canonical_name'] or 'unknown code')}" for x in codes)))
    for kind, note in notes:
        html(ui.callout(note, kind))

    with st.container(border=True):
        st.markdown("**The label, classified** · hover a highlight to see what it is and how it was found")
        html(ui.legend())
        html(ui.tagged_text(result["text"], result["entities"]))


def view_allergens(result: dict):
    allergens = result.get("allergens") or {}
    contains, may = allergens.get("contains", {}), allergens.get("may_contain", {})
    html(ui.section("Allergens", "the 14 major allergen groups (EU 1169/2011, FSSAI)"))
    if not contains and not may:
        html(ui.callout("✅ <b>None of the 14 major allergen groups was found.</b> Always check the packet: "
                        "OCR can miss words.", "good"))
        return
    if contains:
        st.markdown("**Contains**")
        html(ui.allergen_badges(contains, "contains"))
    if may:
        st.markdown("**May contain** (precautionary warning - traces possible)")
        html(ui.allergen_badges(may, "may"))
    html('<div class="muted" style="margin-top:10px">Under each group: the words on the label that triggered it.</div>')


def view_guide(result: dict, rid: str, profile, weight: float, guide: dict):
    html(ui.section("Is this amount OK?", "your portion vs. official daily limits (WHO · EU · EFSA)"))
    c1, c2, c3 = st.columns([1, 1, 2])
    kind = c1.radio("Product type", ["food", "drink"], index=1 if guide["drink"] else 0, horizontal=True,
                    key=f"{rid}-kind")
    unit = "ml" if kind == "drink" else "g"
    portion = c2.number_input(f"Your portion ({unit})", min_value=1.0, max_value=2000.0, value=guide["default_portion"],
                              step=5.0, key=f"{rid}-portion")
    c3.markdown(f'<div class="muted" style="padding-top:30px">Profile: <b>{ui.e(profile.name)}</b>, {weight:g} kg '
                f'- change it in the sidebar.</div>', unsafe_allow_html=True)

    with st.expander("Nutrition values per 100 g / 100 ml  ·  source: " + re.sub(r"\*", "", guide["source"]),
                     expanded=not guide["values"]):
        st.caption("Read from the photo where possible. Correct any number and everything updates.")
        cols = st.columns(4)
        per100 = {}
        for i, key in enumerate(EDITABLE_NUTRIENTS):
            name, nunit = NUTRIENTS[key]
            default = guide["values"].get(key)
            value = cols[i % 4].number_input(f"{name} ({nunit})", min_value=0.0, max_value=900.0 if nunit == "kcal" else 100.0,
                                             value=float(default) if default is not None else None, step=0.1,
                                             format="%.1f", key=f"{rid}-{key}", placeholder="not given")
            if value is not None:
                per100[key] = value

    if not per100:
        html(ui.callout("Type the nutrition values from the packet above to see how a portion compares with daily "
                        "limits.", "info"))
    else:
        rows = assess_portion(per100, portion, profile, kind)
        text, level = headline(rows, portion, unit)
        detail = "Sugar comparison uses the label's total sugars (includes natural milk/fruit sugars) - an upper estimate."
        html(ui.verdict(text, level, detail))
        limited = [r for r in rows if r["limit_kind"] == "max"]
        other = [r for r in rows if r["limit_kind"] != "max"]
        left, right = st.columns(2)
        with left:
            st.markdown("**Nutrients to limit**")
            html("".join(ui.meter(r, unit) for r in limited))
        with right:
            st.markdown("**Energy and nutrients to get enough of**")
            html("".join(ui.meter(r, unit) for r in other))

    for note in special_notes(result["text"] + " " + result.get("ocr", {}).get("full_text", "")):
        html(ui.callout(f"⚠️ <b>{ui.e(note['substance'])}:</b> {ui.e(note['note'])}", "warn"))

    with st.expander("Where do these numbers come from?"):
        st.markdown(
            "- **Sugars:** WHO guideline *Sugars intake for adults and children* (2015): free sugars below 10 % of "
            "energy (≈ 50 g at 2000 kcal), ideally below 5 % (≈ 25 g). ICMR-NIN *Dietary Guidelines for Indians* "
            "(2024) also advise keeping added sugar low.\n"
            "- **Salt:** WHO (2012): less than 5 g salt (2 g sodium) per day for adults, less for young children.\n"
            "- **Fats:** WHO (2023): saturated fat below 10 % and trans fat below 1 % of energy; total fat below 30 %.\n"
            "- **Reference intakes:** EU Regulation 1169/2011, Annex XIII (energy 2000 kcal, fat 70 g, saturates "
            "20 g, carbohydrate 260 g, sugars 90 g, protein 50 g, salt 6 g). Fibre: EFSA (2010), 25 g/day.\n"
            "- **Traffic lights:** UK Food Standards Agency front-of-pack criteria per 100 g / 100 ml.\n"
            "- **Additives:** acceptable daily intakes from JECFA (FAO/WHO) and EFSA scientific opinions.\n\n"
            "General information for healthy people - not personal medical advice.")


def view_additives(result: dict, weight: float):
    additives = [x for x in result["entities"] if x["label"] in ADDITIVE_LABELS]
    html(ui.section("Additives explained", f"{len(additives)} found · daily limits for {weight:g} kg body weight"))
    if not additives:
        html(ui.callout("No additives were found in this ingredient list.", "good"))
        return
    adi_by_number = {}
    for row in additive_guidance(result["entities"], weight):
        number = re.search(r"\(INS ([^)]+)\)$", row["additive"])
        if number:
            adi_by_number[number.group(1)] = row
    seen, cards = set(), []
    for ent in additives:
        key = ent.get("ins_number") or ent["text"].lower()
        if key in seen:
            continue
        seen.add(key)
        cards.append(ui.additive_card(ent, adi_by_number.get(ent.get("ins_number") or "")))
    left, right = st.columns(2)
    with left:
        html("".join(cards[0::2]))
    with right:
        html("".join(cards[1::2]))
    html('<div class="muted">ADI = acceptable daily intake: the amount that can be eaten every day over a lifetime '
         'without appreciable health risk (JECFA / EFSA). Labels rarely state how much additive is inside; permitted '
         'use levels keep normal portions well below the ADI.</div>')


def view_reading(result: dict, active: tuple):
    html(ui.section("How it was read", "the pipeline behind this result"))
    if "ocr" in result:
        o = result["ocr"]
        html(ui.pipeline([f"📷 Photo", f"↻ rotated {o['rotation']}°", f"🔍 enlarged ×{o['scale']}",
                          f"🔤 {o['engine']} · {o['n_lines']} lines", "✏️ spelling fixed",
                          f"✂️ section by {o['section_method']}", f"🧠 {result['ner_model']}", "📚 knowledge base",
                          f"⏱ {o['seconds']} s"]))
        st.write("")
        left, right = st.columns([2, 3])
        with left:
            try:
                import cv2
                st.image(cv2.cvtColor(prepare_photo(active[1], active[2], active[3]), cv2.COLOR_BGR2RGB),
                         caption=f"Photo that was read · mean OCR confidence {o['mean_confidence']:.2f}",
                         width="stretch")
            except Exception:
                pass
        with right:
            st.markdown("**Text read from the photo** - fix any misread word, then re-analyse")
            ocr_text = st.text_area("OCR text", o["full_text"], height=260, label_visibility="collapsed",
                                    key="ocr_" + hashlib.md5(o["full_text"].encode()).hexdigest())
            if st.button("Re-analyse corrected text", key="reanalyse", type="primary"):
                st.session_state.active = ("text", ocr_text)
                st.rerun()
    else:
        html(ui.pipeline(["⌨️ Typed text", f"✂️ section: {result.get('section_method', '')}",
                          f"🧠 {result['ner_model']}", "📚 knowledge base"]))
    with st.expander("Every entity: label, how it was found and how it was linked"):
        st.dataframe(pd.DataFrame(result["entities"])[["text", "label", "source", "canonical_name", "ins_number",
                                                        "function", "link_method", "link_score"]],
                     hide_index=True, width="stretch")


# ------------------------------------------------------------------------------------ page
html(ui.hero())

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    has_model = (FINAL_MODEL_DIR / "config.json").exists()
    options = [name for name, k in CLASSIFIERS.items() if k != "transformer" or has_model]
    choice = st.radio("Classifier", options,
                      help="Hybrid = dictionary rules + fine-tuned DistilBERT + fuzzy matching against the knowledge base.")
    kind = CLASSIFIERS[choice]
    st.markdown("### 👤 Who is eating?")
    profile_name = st.selectbox("Profile", list(PROFILES), label_visibility="collapsed", key="profile")
    profile = PROFILES[profile_name]
    weight = st.number_input("Body weight (kg)", min_value=5.0, max_value=200.0, value=float(profile.body_weight_kg),
                             step=1.0, key=f"weight-{profile_name}")
    st.markdown("### 📸 Photo tips")
    st.markdown("- Fill the frame with the ingredient list (and the nutrition table)\n"
                "- Good light, no glare, hold steady\n- Curved packs: photograph the flat part")
    st.divider()
    st.caption(DISCLAIMER)

if "active" not in st.session_state:
    st.session_state.active = None          # ("image", bytes, rotation, crop) or ("text", str)

active = st.session_state.active
# the input panel folds away once there is a result (same widget, so an uploaded photo and its rotate/crop
# settings stay available when it is opened again)
with st.expander("📥  Scan a product: photo, camera, text or example", expanded=active is None):
    tab_upload, tab_camera, tab_text, tab_example = st.tabs(["🖼️ Upload photo", "📷 Camera", "⌨️ Paste text",
                                                             "📦 Example packets"])
    with tab_upload:
        upload = st.file_uploader("Drop a photo of the ingredient list (JPG / PNG / WEBP)", type=["jpg", "jpeg", "png", "webp"])
        if upload is not None:
            data = upload.getvalue()
            preview_col, tools_col = st.columns([1, 2])
            with tools_col:
                st.markdown("**Rotate / crop** (optional - helps when the list is a small part of the photo)")
                rot_u = st.radio("Rotate", [0, 90, 180, 270], horizontal=True, format_func=lambda a: f"{a}°",
                                 key="rot_upload")
                x_range = st.slider("Keep from left to right (%)", 0, 100, (0, 100), key="crop_x")
                y_range = st.slider("Keep from top to bottom (%)", 0, 100, (0, 100), key="crop_y")
                crop = (x_range[0], x_range[1], y_range[0], y_range[1])
                scan_now = st.button("Scan this photo", type="primary", key="scan_upload")
            with preview_col:
                try:
                    import cv2
                    st.image(cv2.cvtColor(prepare_photo(data, rot_u, crop), cv2.COLOR_BGR2RGB),
                             caption="Photo that will be read", width="stretch")
                except Exception:
                    st.error("This file could not be opened as an image.")
            signature = hashlib.md5(data).hexdigest()
            if st.session_state.get("upload_sig") != signature:          # a new photo -> scan it automatically
                st.session_state.upload_sig = signature
                st.session_state.active = ("image", data, rot_u, crop)
            if scan_now:
                st.session_state.active = ("image", data, rot_u, crop)

    with tab_camera:
        shot = st.camera_input("Point the camera at the ingredient list")
        if shot is not None:
            data = shot.getvalue()
            signature = hashlib.md5(data).hexdigest()
            if st.session_state.get("camera_sig") != signature:
                st.session_state.camera_sig = signature
                st.session_state.active = ("image", data, 0, NO_CROP)

    with tab_text:
        typed = st.text_area("Ingredient list (you can also paste the nutrition table)", EXAMPLE_TEXT, height=130)
        if st.button("Analyse text", type="primary", key="scan_text"):
            st.session_state.active = ("text", typed)

    with tab_example:
        photos = example_photos()
        if photos.empty:
            st.write("No example photos found (run `python -m src.ocr.collect_images`).")
        else:
            pick_col, img_col = st.columns([2, 1])
            title = pick_col.selectbox("60 real packet photos (Open Food Facts, CC BY-SA)", photos["title"].tolist())
            row = photos[photos["title"] == title].iloc[0]
            img_col.image(str(ROOT / row["image"]), width="stretch")
            if pick_col.button("Scan this packet", type="primary", key="scan_example"):
                st.session_state.active = ("image", (ROOT / row["image"]).read_bytes(), 0, NO_CROP)

active = st.session_state.active

if active is None:
    html(ui.features())
    samples = sample_lists()
    if samples:
        st.markdown("**No packet at hand? Try a sample product:**")
        cols = st.columns(len(SAMPLES))
        for col, name in zip(cols, SAMPLES):
            if name in samples and col.button(name, key=f"sample-{name}", width="stretch"):
                st.session_state.active = ("text", samples[name])
                st.rerun()
    st.caption(DISCLAIMER)
    st.stop()

result = None
if active[0] == "image":
    with st.spinner("Reading the label (usually 2-10 seconds) ..."):
        try:
            result = scan_image_cached(active[1], kind, active[2], active[3])
        except Exception as err:
            st.error(f"This file could not be read as an image: {err}")
else:
    result = scan_text_cached(active[1], kind)
if result is None:
    st.stop()

head_left, head_right = st.columns([5, 1])
source = "your photo" if active[0] == "image" else "the typed text"
head_left.markdown(ui.section("Results", f"from {source} · {result['ner_model']}"), unsafe_allow_html=True)
if head_right.button("↺ New scan", key="new_scan", width="stretch"):
    st.session_state.active = None
    st.rerun()

if not result["entities"]:
    html(ui.callout("<b>No ingredients were recognised.</b> Try a sharper, well-lit photo that shows the ingredient "
                    "list, rotate or crop it in the Upload tab, or paste the text.", "bad"))
    if "ocr" in result:
        view_reading(result, active)
    st.stop()

if "ocr" in result and (result["ocr"]["mean_confidence"] < 0.8 or len(result["entities"]) < 3):
    html(ui.callout("<b>This photo is hard to read</b> (blur, curve, glare or very small print), so some ingredients "
                    "may be missing. Crop to the list in the Upload tab, or fix the text in <b>How it was read</b>.",
                    "warn"))

rid = hashlib.md5((result["input_text"] + kind).encode()).hexdigest()[:10]
guide = current_guide(result, rid, profile)
view_summary(result, guide)

t_ing, t_guide, t_all, t_add, t_read = st.tabs(["🧾 Ingredients", "⚖️ Is this amount OK?", "⚠️ Allergens",
                                                 "⚗️ Additives", "🔍 How it was read"])
with t_ing:
    view_ingredients(result)
with t_guide:
    view_guide(result, rid, profile, weight, guide)
with t_all:
    view_allergens(result)
with t_add:
    view_additives(result, weight)
with t_read:
    view_reading(result, active)

st.caption(DISCLAIMER)
