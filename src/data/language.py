"""A small, explainable language identifier for ingredient lists.

Why not a library such as langdetect/fastText?
  * Ingredient lists are short and full of chemical names, which general-purpose
    detectors often get wrong.
  * We only need one decision: "is this list written in English?".
  * A rule we can explain in one sentence is easier to defend in a viva.

How it works:
  1. Each language has a list of very common ingredient-label words
     (e.g. English "sugar", "salt", "flour"; French "sucre", "sel", "farine").
  2. Words that appear in more than one list ("cacao", "aroma", "natural", "de")
     are removed automatically, so only language-EXCLUSIVE marker words count.
  3. We count marker words in the text. The English score is
         english_score = english_hits / all_marker_hits
     A bilingual Canadian label ("sugar, salt / sucre, sel") scores about 0.5.
  4. Texts with no marker words at all get language "unknown".
"""
import re
from collections import Counter

_MARKERS = {
    "en": """and of with contains contain may less than or from sugar salt flour wheat milk oil
        vegetable flavour flavor flavouring flavoring powder extract acid sodium starch corn syrup
        cocoa cream skimmed skim whole dried egg eggs soy soya lecithin yeast spices spice rice
        juice concentrate tomato tomatoes onion onions garlic pepper black cheese chicken beef pork
        peanuts peanut nuts almonds emulsifier emulsifiers preservative preservatives colour color
        colours colors acidity regulator raising agent agents stabiliser stabilizer thickener
        antioxidant sweetener citric ascorbic iron calcium potassium chloride bicarbonate refined
        palm sunflower rapeseed canola coconut edible modified maize gum glucose fructose dextrose
        the for ingredients made traces flakes seeds honey vinegar mustard lemon orange apple
        strawberry banana potato potatoes beans peas lentils chickpeas oats barley cultures enzymes
        whey gelatin artificial sauce paste chilli chili ginger turmeric cumin coriander cardamom
        cinnamon cloves water butter natural""",
    "fr": """et du des la le les sucre sel eau farine blé ble lait huile beurre oeufs œufs arôme
        arome arômes naturel naturels naturelle acidifiant conservateur émulsifiant emulsifiant
        colorant épaississant epaississant amidon poudre sirop jus levure crème creme fromage poivre
        peut contenir graines viande poulet porc tomates oignon oignons ail avec issu issus matière
        matiere grasse végétale vegetale végétales noisettes chocolat citron vinaigre de""",
    "de": """und zucker salz wasser mehl weizenmehl milch öl eier natürliches natürliche aroma
        säuerungsmittel konservierungsstoff emulgator farbstoff verdickungsmittel stärke pulver sirup
        saft hefe sahne käse pfeffer kann spuren enthalten von mit aus der die das gewürze zwiebeln
        knoblauch schweinefleisch kakao zitronensaft essig pflanzliches pflanzliche rapsöl
        sonnenblumenöl vollmilchpulver magermilchpulver butter""",
    "es": """y azúcar azucar sal agua harina trigo leche aceite mantequilla huevo huevos aroma
        acidulante conservador conservante emulgente colorante espesante almidón almidon polvo
        jarabe zumo jugo levadura nata queso pimienta puede contener trazas con cebolla ajo pollo
        cerdo cacao limón limon vinagre girasol vegetal especias natural de""",
    "it": """e di zucchero sale acqua farina frumento latte olio burro uova aromi aroma naturali
        acidificante conservante emulsionante emulsionanti colorante addensante amido polvere
        sciroppo succo lievito panna formaggio pepe può contenere tracce con cipolla aglio pollo
        maiale cacao limone aceto girasole vegetale spezie grano tenero""",
    "nl": """en suiker zout bloem tarwebloem melk olie boter eieren aroma zuurteregelaar
        conserveermiddel emulgator kleurstof verdikkingsmiddel zetmeel poeder stroop sap gist room
        kaas peper kan sporen bevatten van met uien knoflook kip varkensvlees cacao citroen azijn
        zonnebloemolie plantaardige kruiden specerijen water""",
    "pt": """e açúcar acucar sal água agua farinha trigo leite óleo oleo manteiga ovos aroma
        acidulante conservante emulsionante corante espessante amido pó xarope suco sumo fermento
        nata queijo pimenta pode conter vestígios cebola alho frango porco cacau limão vinagre
        girassol vegetal especiarias de""",
}


def _build_exclusive_markers(markers: dict) -> dict:
    word_sets = {lang: set(words.split()) for lang, words in markers.items()}
    counts = Counter(word for words in word_sets.values() for word in words)
    return {lang: {w for w in words if counts[w] == 1} for lang, words in word_sets.items()}


EXCLUSIVE_MARKERS = _build_exclusive_markers(_MARKERS)
_WORD_RE = re.compile(r"[^\W\d_]+")  # sequences of letters (any alphabet), no digits/underscores


def language_scores(text: str) -> Counter:
    """Count language-exclusive marker words per language."""
    hits = Counter()
    for word in _WORD_RE.findall(str(text).lower()):
        for lang, words in EXCLUSIVE_MARKERS.items():
            if word in words:
                hits[lang] += 1
    return hits


def detect_language(text: str) -> tuple:
    """Return (language, english_score). language is 'unknown' when no marker word is found."""
    hits = language_scores(text)
    total = sum(hits.values())
    if total == 0:
        return "unknown", 0.0
    language = hits.most_common(1)[0][0]
    return language, hits["en"] / total
