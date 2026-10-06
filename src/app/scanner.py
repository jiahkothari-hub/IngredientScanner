"""The complete pipeline: packet photo (or text) -> understandable result.

    PACKET IMAGE -> OCR -> OCR code clean-up -> INGREDIENTS SECTION -> TEXT NORMALISATION
    -> NER MODEL (DistilBERT, or dictionary rules as fallback) -> ENTITY LINKING (knowledge base)
    -> CATEGORY SUMMARY -> USER-FRIENDLY RESULT

    from src.app.scanner import IngredientScanner
    scanner = IngredientScanner()                    # uses the fine-tuned model if it is present
    result = scanner.scan_image("data/images/packets/8901063162518.jpg")
    result = scanner.scan_text("Sugar, Acidity regulator (INS 330), palm oil")

The result describes what each ingredient IS and what it is USED FOR. It never says whether a
product is healthy or not (identification and interpretation, no health claims).
"""
import re
import time

from src.kb.entity_linking import get_linker
from src.labeling.interpret import declared_classes
from src.utils.config import project_path

FINAL_MODEL_DIR = project_path("models/ingredient-ner-distilbert")
CATEGORY_ORDER = ["SUGAR", "SWEETENER", "FAT", "PRESERVATIVE", "COLOUR", "ADDITIVE", "FLAVOURING", "INGREDIENT"]
CATEGORY_TITLES = {"SUGAR": "Sugars", "SWEETENER": "Sweeteners", "FAT": "Fats & oils",
                   "PRESERVATIVE": "Preservatives", "COLOUR": "Colours", "ADDITIVE": "Other additives",
                   "FLAVOURING": "Flavourings", "INGREDIENT": "Other ingredients"}
DISCLAIMER = ("This tool identifies ingredients and explains what they are and what they are used for. "
              "It does not assess whether a product is healthy or safe. OCR and automatic classification "
              "can make mistakes: always check the original label.")


def load_ner(kind: str = "auto"):
    """'transformer' = fine-tuned DistilBERT, 'dictionary' = rule baseline, 'auto' = model if available."""
    if kind in ("auto", "transformer") and (FINAL_MODEL_DIR / "config.json").exists():
        from src.ner.transformer_ner import TransformerTagger
        return TransformerTagger(FINAL_MODEL_DIR)
    if kind == "transformer":
        raise FileNotFoundError(f"No fine-tuned model in {FINAL_MODEL_DIR}")
    from src.baseline.dictionary_ner import DictionaryNER
    ner = DictionaryNER()
    ner.name = "dictionary_rules"
    return ner


class IngredientScanner:
    def __init__(self, ner: str = "auto"):
        self.ner = load_ner(ner)
        self.ner_name = getattr(self.ner, "name", "model")
        self.linker = get_linker()

    def scan_image(self, image) -> dict:
        """image: file path, bytes or OpenCV array."""
        from src.ocr.ocr_cleanup import fix_additive_codes
        from src.ocr.ocr_pipeline import read_image
        from src.ocr.section_extraction import extract_ingredients_section
        start = time.time()
        ocr = read_image(image)
        cleaned = fix_additive_codes(ocr["text"])
        section = extract_ingredients_section(cleaned)
        result = self.scan_text(section["text"])
        result["ocr"] = {"full_text": ocr["text"], "corrected_text": cleaned, "section_method": section["method"],
                         "rotation": ocr["rotation"], "mean_confidence": ocr["mean_confidence"],
                         "seconds": round(time.time() - start, 1)}
        return result

    def scan_text(self, text: str) -> dict:
        prediction = self.ner.predict(text)
        norm_text, entities = prediction["text"], prediction["entities"]
        declared = declared_classes(norm_text, entities)
        items = []
        for i, ent in enumerate(entities):
            link = self.linker.link(ent)
            function = link["function"]
            if i in declared and declared[i]:
                function = declared[i].replace("-", " ")      # what THIS label says it is used for
            items.append({"text": ent["text"], "label": ent["label"], "start": ent["start"], "end": ent["end"],
                          **link, "function": function, "function_source": "label" if i in declared else
                          ("reference" if function else "")})
        return {"input_text": text, "text": norm_text, "entities": items, "summary": summarise(items),
                "ner_model": self.ner_name, "disclaimer": DISCLAIMER}


def summarise(items: list) -> dict:
    """Group entities into the categories shown to the user."""
    groups = {c: [] for c in CATEGORY_ORDER}
    for it in items:
        label = it["label"]
        if label == "INS_CODE":                   # a code is reported under the category of its additive
            label = it["kb_category"] if it["kb_category"] in groups else "ADDITIVE"
            if label in ("INS_CODE", "INGREDIENT"):
                label = "ADDITIVE"
        if label == "FUNCTION_CLASS":
            continue
        name = it["text"]
        if it["label"] == "INS_CODE" and it["canonical_name"]:
            name = f"{it['text']} ({it['canonical_name']})"
        if name.lower() not in {n.lower() for n in groups[label]}:
            groups[label].append(name)
    # "hidden" names: sugars whose name does not contain the word sugar, fats without oil/fat/butter,
    # and additives written only as a code
    hidden = {
        "sugars_under_other_names": [n for n in groups["SUGAR"] if not re.search(r"sugar", n, re.I)],
        "fats_under_other_names": [n for n in groups["FAT"] if not re.search(r"oil|fat|butter|ghee", n, re.I)],
        "additives_written_as_codes": [it["text"] for it in items if it["label"] == "INS_CODE"],
    }
    return {"counts": {c: len(v) for c, v in groups.items()}, "groups": groups, "hidden": hidden}
