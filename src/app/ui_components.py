"""Visual building blocks for the Streamlit app: theme CSS and small HTML components.

Kept separate from the page logic (streamlit_app.py) so the look can be changed in one place. Every
component returns an HTML string that the page renders with st.markdown(..., unsafe_allow_html=True);
all user/label text is escaped with html.escape before it is inserted.
"""
import html

# category -> (colour, soft background, icon, plural title)
CATEGORY_STYLE = {
    "SUGAR": ("#E8590C", "#FFF0E6", "🍬", "Sugars"),
    "SWEETENER": ("#D6336C", "#FFEEF4", "🧪", "Sweeteners"),
    "FAT": ("#E09200", "#FFF6E0", "🧈", "Fats & oils"),
    "PRESERVATIVE": ("#E03131", "#FFEDED", "🛡️", "Preservatives"),
    "COLOUR": ("#7048E8", "#F2EEFF", "🎨", "Colours"),
    "ADDITIVE": ("#1C7ED6", "#EAF4FF", "⚗️", "Other additives"),
    "FLAVOURING": ("#2F9E44", "#EBF8EE", "🌿", "Flavourings"),
    "INGREDIENT": ("#5C677D", "#F1F3F6", "🥣", "Other ingredients"),
}
LABEL_COLOURS = {**{k: v[0] for k, v in CATEGORY_STYLE.items()}, "INS_CODE": "#0CA678", "FUNCTION_CLASS": "#495057"}
LABEL_NAMES = {"SUGAR": "Sugar", "SWEETENER": "Sweetener", "FAT": "Fat / oil", "PRESERVATIVE": "Preservative",
               "COLOUR": "Colour", "ADDITIVE": "Additive", "INS_CODE": "INS / E code",
               "FUNCTION_CLASS": "Function on label", "FLAVOURING": "Flavouring", "INGREDIENT": "Ingredient"}
LEVEL_STYLE = {   # share of a daily limit -> colour
    "ok": ("#2F9E44", "#EBF8EE"), "caution": ("#E09200", "#FFF6E0"), "high": ("#E03131", "#FFEDED"),
    "unknown": ("#5C677D", "#F1F3F6"),
}
LIGHT_STYLE = {"low": ("#2F9E44", "Low"), "medium": ("#E09200", "Medium"), "high": ("#E03131", "High")}

e = html.escape

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root { --brand:#0E7C66; --brand-dark:#0A5C4C; --ink:#1B2430; --muted:#5C677D; --line:#E4E8EC;
        --card:#FFFFFF; --soft:#F6F8F7; }
html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea {
  font-family: 'Plus Jakarta Sans', system-ui, -apple-system, 'Segoe UI', sans-serif !important; }
.stApp { background: linear-gradient(180deg, #F3F8F6 0px, #F7F8FA 420px); }
.block-container { padding-top: 3.4rem; max-width: 1180px; }
header[data-testid="stHeader"] { background: transparent; }
h1, h2, h3, h4 { color: var(--ink); letter-spacing: -0.01em; }

/* hero */
.hero { background: radial-gradient(1200px 300px at 10% -20%, #1FB58F 0%, transparent 60%),
        linear-gradient(120deg, #0A5C4C 0%, #0E7C66 55%, #13967A 100%);
        border-radius: 22px; padding: 30px 34px 26px; color: #fff; position: relative; overflow: hidden;
        box-shadow: 0 14px 40px -18px rgba(10, 92, 76, .55); margin-bottom: 18px; }
.hero:after { content: "🥗"; position: absolute; right: 26px; top: 14px; font-size: 92px; opacity: .16; }
.hero h1 { color: #fff; font-size: 2.15rem; font-weight: 800; margin: 0 0 6px; padding: 0; }
.hero p { color: rgba(255,255,255,.88); font-size: 1.02rem; margin: 0 0 16px; max-width: 760px; }
.steps { display: flex; flex-wrap: wrap; gap: 8px; }
.step { background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.22); border-radius: 999px;
        padding: 5px 13px; font-size: .84rem; font-weight: 600; color: #fff; }
.step b { background: #fff; color: var(--brand-dark); border-radius: 50%; display: inline-block; width: 19px;
          height: 19px; line-height: 19px; text-align: center; font-size: .72rem; margin-right: 6px; }

/* generic cards */
.card { background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 18px 20px;
        box-shadow: 0 1px 2px rgba(16,24,40,.04); margin-bottom: 14px; }
.card h4 { margin: 0 0 4px; font-size: 1.02rem; }
.section-title { display: flex; align-items: baseline; flex-wrap: wrap; gap: 4px 12px; margin: 8px 0 12px; }
.section-title .h { margin: 0; font-size: 1.35rem; font-weight: 800; color: var(--ink); letter-spacing: -0.01em; }
.section-title span { color: var(--muted); font-size: .9rem; }
.muted { color: var(--muted); font-size: .88rem; }

/* landing feature cards */
.features { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 4px 0 6px; }
.feature { background: #fff; border: 1px solid var(--line); border-radius: 16px; padding: 16px 18px; }
.feature .ic { font-size: 1.6rem; }
.feature h4 { margin: 6px 0 4px; font-size: 1rem; }
.feature p { margin: 0; color: var(--muted); font-size: .88rem; line-height: 1.45; }

/* at a glance */
.glance { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 6px 0 16px; }
.tile { background: #fff; border: 1px solid var(--line); border-radius: 16px; padding: 14px 16px;
        border-top: 4px solid var(--c); }
.tile .k { color: var(--muted); font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; }
.tile .v { color: var(--ink); font-size: 1.65rem; font-weight: 800; line-height: 1.2; margin-top: 2px; }
.tile .s { color: var(--muted); font-size: .84rem; margin-top: 2px; }

/* category cards */
.cats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 14px; }
.cat { background: var(--bg); border-radius: 16px; padding: 13px 14px; border: 1px solid rgba(0,0,0,.04); min-height: 108px; }
.cat .top { display: flex; align-items: center; justify-content: space-between; }
.cat .name { font-weight: 700; color: var(--c); font-size: .9rem; }
.cat .n { font-weight: 800; font-size: 1.5rem; color: var(--c); }
.cat .items { margin-top: 6px; display: flex; flex-wrap: wrap; gap: 5px; }
.cat .none { color: #98A2B3; font-size: .82rem; margin-top: 8px; }
.chip { background: #fff; color: var(--ink); border: 1px solid rgba(0,0,0,.07); border-radius: 999px;
        padding: 2px 9px; font-size: .78rem; font-weight: 500; }

/* classified text */
.legend { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.lg { font-size: .74rem; font-weight: 700; color: #fff; padding: 2px 9px; border-radius: 999px; }
.tagged { line-height: 2.45; font-size: 1rem; color: var(--ink); }
.ent { color: #fff; padding: 2px 7px; border-radius: 7px; white-space: nowrap; font-weight: 600; }
.ent sub { font-size: .6em; opacity: .9; font-weight: 600; margin-left: 3px; text-transform: lowercase; }

/* callouts */
.callout { border-radius: 14px; padding: 12px 15px; margin: 8px 0; border: 1px solid; font-size: .93rem; }
.callout b { font-weight: 700; }

/* verdict */
.verdict { border-radius: 18px; padding: 18px 20px; display: flex; gap: 16px; align-items: center;
           border: 1px solid; margin: 4px 0 16px; }
.verdict .em { font-size: 2.2rem; }
.verdict .t { font-weight: 800; font-size: 1.08rem; color: var(--ink); }
.verdict .d { color: var(--muted); font-size: .9rem; margin-top: 2px; }

/* meters */
.meter { background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 12px 15px; margin-bottom: 10px; }
.meter .row { display: flex; justify-content: space-between; align-items: baseline; gap: 10px; }
.meter .nm { font-weight: 700; color: var(--ink); }
.meter .val { font-weight: 800; color: var(--ink); }
.meter .val small { color: var(--muted); font-weight: 500; }
.bar { position: relative; height: 10px; background: #EEF1F4; border-radius: 999px; margin: 8px 0 7px; overflow: hidden; }
.bar .fill { height: 100%; border-radius: 999px; }
.bar .tick { position: absolute; top: -2px; bottom: -2px; width: 2px; background: rgba(27,36,48,.35); }
.meter .ft { display: flex; flex-wrap: wrap; gap: 6px 14px; color: var(--muted); font-size: .82rem; }
.pill { display: inline-block; border-radius: 999px; padding: 1px 9px; font-size: .74rem; font-weight: 700; color: #fff; }

/* additive cards */
.add { background: #fff; border: 1px solid var(--line); border-left: 5px solid var(--c); border-radius: 14px;
       padding: 12px 15px; margin-bottom: 10px; }
.add .hd { display: flex; justify-content: space-between; gap: 8px; align-items: center; }
.add .nm { font-weight: 700; color: var(--ink); }
.add .code { font-size: .74rem; font-weight: 800; color: var(--c); background: var(--bg); border-radius: 8px; padding: 2px 8px; white-space: nowrap; }
.add .fn { font-size: .84rem; color: var(--ink); margin-top: 3px; }
.add .ds { font-size: .82rem; color: var(--muted); margin-top: 4px; line-height: 1.45; }
.add .adi { font-size: .8rem; margin-top: 7px; padding-top: 7px; border-top: 1px dashed var(--line); color: var(--ink); }

/* allergens */
.alg { display: flex; flex-wrap: wrap; gap: 8px; }
.alg .a { border-radius: 12px; padding: 8px 12px; font-weight: 700; font-size: .9rem; border: 1px solid; }
.alg .a small { display: block; font-weight: 500; font-size: .76rem; opacity: .85; }

/* pipeline */
.pipe { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.pipe .p { background: #fff; border: 1px solid var(--line); border-radius: 10px; padding: 6px 10px; font-size: .8rem; font-weight: 600; }
.pipe .arrow { color: #98A2B3; }

/* streamlit widgets */
.stTabs [role="tablist"] { gap: 6px; background: #EAF0ED; padding: 5px; border-radius: 14px; border: none; }
.stTabs [role="tablist"] > :not([role="tab"]) { display: none !important; }     /* underline indicator */
.stTabs [role="tab"] { border-radius: 10px !important; padding: 8px 16px !important; border: none !important;
                       margin: 0 !important; transition: background .15s; }
.stTabs [role="tab"]::after, .stTabs [role="tab"]::before { display: none !important; }
.stTabs [role="tab"]:hover { background: rgba(255,255,255,.6); }
.stTabs [role="tab"] p { font-weight: 700; font-size: .92rem; }
.stTabs [role="tab"][aria-selected="true"] { background: #fff !important; box-shadow: 0 1px 4px rgba(16,24,40,.14); }
.stTabs [role="tab"][aria-selected="true"] p { color: var(--brand-dark); }
div[data-testid="stFileUploaderDropzone"] { border-radius: 16px; border: 2px dashed #9CC9BC; background: #F4FBF8; }
.stButton > button { border-radius: 12px; font-weight: 700; }
section[data-testid="stSidebar"] { background: #FBFCFB; border-right: 1px solid var(--line); }
div[data-testid="stExpander"] details { border-radius: 16px; background: #fff; border: 1px solid var(--line); }
div[data-testid="stExpander"] summary p { font-weight: 700; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px; }
@media (max-width: 900px) { .glance, .cats, .features { grid-template-columns: repeat(2, 1fr); } .hero h1 { font-size: 1.6rem; } }
@media (max-width: 520px) { .glance, .cats, .features { grid-template-columns: 1fr; } }
</style>
"""


def hero() -> str:
    steps = ["Snap the label", "We read it (OCR)", "AI classifies every ingredient", "See what it means for you"]
    pills = "".join(f'<span class="step"><b>{i}</b>{e(s)}</span>' for i, s in enumerate(steps, 1))
    return (f'<div class="hero"><h1>Ingredient Scanner</h1>'
            f'<p>Point your camera at any food label. Find hidden sugars, fats, additives and E/INS codes, '
            f'spot allergens, and see how much of your daily limit a portion uses.</p>'
            f'<div class="steps">{pills}</div></div>')


def section(title: str, subtitle: str = "") -> str:
    return f'<div class="section-title"><div class="h">{e(title)}</div><span>{e(subtitle)}</span></div>'


def features() -> str:
    items = [("🔎", "Hidden names, decoded", "Dextrose, invert syrup and maltodextrin are sugars by another name. "
              "INS 330 is citric acid. The app tells you."),
             ("⚠️", "Allergens at a glance", "The 14 major allergen groups, with a separate list for "
              "\"may contain\" warnings."),
             ("⚖️", "Is this amount OK?", "Your portion compared with WHO and EU daily limits, plus the "
              "acceptable daily intake of every additive.")]
    return '<div class="features">' + "".join(
        f'<div class="feature"><div class="ic">{i}</div><h4>{e(t)}</h4><p>{e(d)}</p></div>' for i, t, d in items) + "</div>"


def glance(tiles: list) -> str:
    """tiles: [(label, value, sub, colour)]"""
    return '<div class="glance">' + "".join(
        f'<div class="tile" style="--c:{c}"><div class="k">{e(k)}</div><div class="v">{e(str(v))}</div>'
        f'<div class="s">{e(s)}</div></div>' for k, v, s, c in tiles) + "</div>"


def category_cards(groups: dict, order: list) -> str:
    cards = []
    for cat in order:
        colour, bg, icon, title = CATEGORY_STYLE[cat]
        items = groups.get(cat, [])
        chips = ("".join(f'<span class="chip">{e(x)}</span>' for x in items[:12])
                 + (f'<span class="chip">+{len(items) - 12} more</span>' if len(items) > 12 else "")
                 if items else '<div class="none">None found</div>')
        cards.append(f'<div class="cat" style="--c:{colour};--bg:{bg}"><div class="top"><span class="name">'
                     f'{icon} {e(title)}</span><span class="n">{len(items)}</span></div>'
                     f'<div class="items">{chips}</div></div>')
    return '<div class="cats">' + "".join(cards) + "</div>"


def legend() -> str:
    return '<div class="legend">' + "".join(
        f'<span class="lg" style="background:{LABEL_COLOURS[l]}">{e(LABEL_NAMES[l])}</span>'
        for l in ["SUGAR", "SWEETENER", "FAT", "PRESERVATIVE", "COLOUR", "ADDITIVE", "INS_CODE", "FLAVOURING",
                  "FUNCTION_CLASS", "INGREDIENT"]) + "</div>"


def tagged_text(text: str, entities: list) -> str:
    parts, pos = [], 0
    for ent in entities:
        if ent["start"] < pos:
            continue
        parts.append(e(text[pos:ent["start"]]))
        tip = LABEL_NAMES[ent["label"]] + (f" - {ent['canonical_name']}" if ent.get("canonical_name") else "")
        tip += f" (found by: {ent.get('source', '')})"
        parts.append(f'<span class="ent" title="{e(tip)}" style="background:{LABEL_COLOURS[ent["label"]]}">'
                     f'{e(text[ent["start"]:ent["end"]])}<sub>{e(LABEL_NAMES[ent["label"]])}</sub></span>')
        pos = ent["end"]
    parts.append(e(text[pos:]))
    return '<div class="tagged">' + "".join(parts) + "</div>"


def callout(text_html: str, kind: str = "info") -> str:
    colours = {"info": ("#1C7ED6", "#EAF4FF"), "warn": ("#B76E00", "#FFF6E0"), "bad": ("#C92A2A", "#FFEDED"),
               "good": ("#2B8A3E", "#EBF8EE")}
    c, bg = colours[kind]
    return f'<div class="callout" style="color:{c};background:{bg};border-color:{c}33">{text_html}</div>'


def verdict(text: str, level: str, detail: str = "") -> str:
    c, bg = LEVEL_STYLE.get(level, LEVEL_STYLE["unknown"])
    icon = {"ok": "✅", "caution": "⚠️", "high": "🛑"}.get(level, "ℹ️")
    return (f'<div class="verdict" style="background:{bg};border-color:{c}55"><div class="em">{icon}</div>'
            f'<div><div class="t">{e(text)}</div><div class="d">{e(detail)}</div></div></div>')


def share_level(share: float) -> str:
    return "high" if share >= 0.5 else "caution" if share >= 0.25 else "ok"


def amount(rounded: float, exact: float = None) -> str:
    """Readable amount: small values keep two decimals (0.03 g salt, not 0 g)."""
    value = exact if exact is not None else rounded
    if value == 0:
        return "0"
    if value < 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.1f}".rstrip("0").rstrip(".")


def meter(row: dict, unit_word: str) -> str:
    """A daily-limit bar: fill = share of the day's limit used by the portion."""
    share = row["share"]
    colour = LEVEL_STYLE[share_level(share)][0] if row["limit_kind"] == "max" else "#0E7C66"
    width = min(share, 1.0) * 100
    foot = [f"<span><b>{share:.0%}</b> of the daily {'limit' if row['limit_kind'] == 'max' else 'reference'}</span>"]
    if row.get("traffic_light") in LIGHT_STYLE:
        lc, lt = LIGHT_STYLE[row["traffic_light"]]
        foot.append(f'<span>per 100 {unit_word}: <span class="pill" style="background:{lc}">{lt}</span></span>')
    if row.get("max_product_g"):
        foot.append(f"<span>{row['max_product_g']:,} {unit_word} would use the whole day's limit</span>")
    row.setdefault("value_exact", None)
    if row["key"] in ("sugars_g", "added_sugars_g") and row["in_portion"] > 0:
        foot.append(f"<span>≈ {row['in_portion'] / 4:.1f} teaspoons</span>")
    return (f'<div class="meter" title="{e(row["source"])}"><div class="row"><span class="nm">{e(row["nutrient"])}</span>'
            f'<span class="val">{amount(row["in_portion"], row["value_exact"])} {e(row["unit"])} <small>of {row["daily_value"]:g} {e(row["unit"])}'
            f'</small></span></div><div class="bar"><div class="fill" style="width:{width:.1f}%;background:{colour}"></div>'
            f'</div><div class="ft">{"".join(foot)}</div></div>')


def additive_card(ent: dict, adi: dict = None) -> str:
    cat = ent["label"] if ent["label"] in CATEGORY_STYLE else (ent.get("kb_category") or "ADDITIVE")
    colour, bg = CATEGORY_STYLE.get(cat, CATEGORY_STYLE["ADDITIVE"])[:2]
    name = ent.get("canonical_name") or ent["text"]
    code = f"INS {ent['ins_number']}" if ent.get("ins_number") else "no code"
    function = ent.get("function") or ""
    fn = (f'<div class="fn">Used as <b>{e(function)}</b>{" (stated on the label)" if ent.get("function_source") == "label" else ""}'
          f' · on label: “{e(ent["text"])}”</div>') if function else f'<div class="fn">On label: “{e(ent["text"])}”</div>'
    desc = f'<div class="ds">{e(ent.get("description") or "")}</div>' if ent.get("description") else ""
    adi_html = ""
    if adi:
        adi_html = f'<div class="adi">⚖️ <b>Daily limit (ADI):</b> {e(adi["daily_limit"])} <span class="muted">· {e(adi["authority"])}</span>'
        if adi.get("note"):
            adi_html += f'<br><span class="muted">{e(adi["note"])}</span>'
        adi_html += "</div>"
    return (f'<div class="add" style="--c:{colour};--bg:{bg}"><div class="hd"><span class="nm">{e(name)}</span>'
            f'<span class="code">{e(code)}</span></div>{fn}{desc}{adi_html}</div>')


def allergen_badges(groups: dict, kind: str) -> str:
    c, bg = ("#C92A2A", "#FFEDED") if kind == "contains" else ("#B76E00", "#FFF6E0")
    return '<div class="alg">' + "".join(
        f'<div class="a" style="color:{c};background:{bg};border-color:{c}44">{e(g)}<small>{e(", ".join(w))}</small></div>'
        for g, w in groups.items()) + "</div>"


def pipeline(steps: list) -> str:
    out = []
    for i, s in enumerate(steps):
        if i:
            out.append('<span class="arrow">→</span>')
        out.append(f'<span class="p">{e(s)}</span>')
    return '<div class="pipe">' + "".join(out) + "</div>"
