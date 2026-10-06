"""Build notebooks/ingredient_scanner_person2_person3.ipynb (run from the project root).

    python notebooks/build_person2_person3_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/ingredient_scanner_person2_person3.ipynb
"""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))


def explain(what, why, inp, out, viva):
    md(f"""> **WHAT WE DID:** {what}
>
> **WHY:** {why}
>
> **INPUT → OUTPUT:** {inp} → {out}
>
> **HOW TO EXPLAIN IT IN THE VIVA:** *"{viva}"*""")


md("""
# Ingredient Scanner: Person 2 (Deep Learning NER) and Person 3 (OCR, knowledge base, app)

This notebook continues `ingredient_scanner_person1_pipeline.ipynb`, which built the data foundation.
Here we:
- train and compare NER models (dictionary rules, CRF, DistilBERT, DistilBERT with OCR-noise augmentation, DistilBERT + CRF, BERT-base);
- build the OCR pipeline, knowledge base and entity linking;
- connect everything into the photo-to-result app.

```
PACKET IMAGE → OCR (EasyOCR) → code clean-up → INGREDIENTS SECTION → NORMALISATION
→ NER (DistilBERT) → ENTITY LINKING (knowledge base) → CATEGORY SUMMARY → USER-FRIENDLY RESULT
```

**Honesty note.**
- All models are trained on **silver** (rule-made) labels.
- *silver test* scores measure how well a model learned the labelling policy.
- *noisy test* scores measure robustness to OCR-style errors, the realistic setting.
- The *real photos* evaluation measures the full pipeline on 60 real packet photos.
- **No human-verified gold labels exist yet.** Gold scores appear automatically after annotation.
""")
code("""
import os, sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from IPython.display import Image, Markdown, HTML, display
pd.set_option("display.max_colwidth", 110)
def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
""")

# ------------------------------------------------------------------ Person 2
md("""
## Part A: Person 2, Deep Learning NER

### A1. Pretrained models

| Model | Parameters | Source |
|---|---|---|
| `distilbert-base-uncased` | 66 M | Sanh et al. 2019, Hugging Face's original public S3 bucket |
| `bert-base-cased` | 108 M | Devlin et al. 2019, same source |

`src/ner/pretrained.py` downloads them with plain HTTPS. The project therefore also works where huggingface.co is blocked.
""")
md("### A2. From word labels to sub-word labels")
code("""
from transformers import AutoTokenizer
from src.ner.transformer_ner import encode, IGNORE
from src.labeling.bio import ID2LABEL
tok = AutoTokenizer.from_pretrained("models/ingredient-ner-distilbert")
words = ["Maltodextrin", ",", "Acidity", "regulator", "(", "INS", "330", ")"]
tags  = ["B-INGREDIENT", "O", "B-FUNCTION_CLASS", "I-FUNCTION_CLASS", "O", "B-INS_CODE", "I-INS_CODE", "O"]
item = encode(tok, words, tags)
pd.DataFrame({"sub-word": tok.convert_ids_to_tokens(item["input_ids"]),
              "training label": [ID2LABEL[l] if l != IGNORE else "-100 (ignored)" for l in item["labels"]]}).T
""")
explain("Each word's first sub-word gets the word's label; other sub-words and the special tokens [CLS]/[SEP] get -100, which the loss ignores.",
        "The Transformer predicts one label per sub-word, but our data and evaluation are per word.",
        "word tokens + BIO tags", "sub-word ids + aligned label ids",
        "We follow the standard Hugging Face token-classification recipe and predict each word from its first sub-word, so scores stay comparable at word level.")

md("### A3. Training runs (same seeded 8,000 silver sentences for every learned model)")
code("""
rows = []
for log_path in sorted(Path("models/runs").glob("*/best/training_log.json")):
    log = load(log_path); a = log["args"]
    rows.append({"run": log_path.parent.parent.name, "base model": a["model"], "CRF head": a["crf"], "OCR augmentation": a["augment"],
                 "epochs": a["epochs"], "lr": a["lr"], "train sentences": a["max_train"],
                 "val F1 per epoch": [e["val_f1"] for e in log["epochs"]], "minutes": log["epochs"][-1]["minutes"]})
crf_info = Path("models/crf/training_info.json")
if crf_info.exists():
    info = load(crf_info)
    rows.append({"run": "crf", "base model": "none (hand-made features)", "CRF head": True, "OCR augmentation": False,
                 "epochs": f"{info['max_iterations']} L-BFGS iterations", "lr": "-", "train sentences": info["train_sentences"],
                 "val F1 per epoch": "-", "minutes": round(info["seconds"] / 60, 1)})
pd.DataFrame(rows)
""")
md("""
Hyper-parameters (`src/ner/transformer_ner.py`):
- AdamW, weight decay 0.01;
- linear warm-up (10%) then decay; gradient clipping 1.0;
- batch 16, max 256 sub-words, length-bucketed batches;
- best epoch chosen on silver validation.

Training ran on a 4-core CPU with no GPU, which is why we use 8,000 sentences and 2 epochs for every model.
""")

md("### A4. Comparison: Dictionary vs CRF vs DistilBERT vs BERT")
code("""
comparison = load("reports/model_comparison.json")
summary = pd.DataFrame(comparison["summary_f1"]).T
display(summary.round(3))
display(Image("reports/figures/12_model_comparison.png", width=900))
display(Image("reports/figures/13_per_class_f1_noisy.png", width=800))
""")
md("""
How to read this:
* **silver_test.** The dictionary is 1.0 by construction, because silver labels *are* its output. A learned
  model close to 1.0 has learned the labelling policy and generalises it to unseen products.
* **noisy_test.** Same products with OCR-style character errors. The dictionary drops sharply because exact
  string matching fails. Models that learned from context hold up better, and the OCR-augmented model is
  trained for exactly this.
* **gold_test.** The real benchmark, available after the team's annotation.
""")
code("""
details = comparison["details"]
best = max((m for m in details if m != "dictionary"), key=lambda m: details[m]["noisy_test_0.05"]["micro"]["f1"])
from src.evaluation.metrics import format_report
print("Best system on noisy text:", best)
display(Markdown(format_report(details[best]["noisy_test_0.05"], f"{best} on noisy_test_0.05")))
display(Markdown(format_report(details["dictionary"]["noisy_test_0.05"], "dictionary on noisy_test_0.05")))
""")

md("### A5. Side by side on OCR-like text")
code("""
from src.baseline.dictionary_ner import DictionaryNER
from src.ner.transformer_ner import TransformerTagger
model = TransformerTagger("models/ingredient-ner-distilbert")
rules = DictionaryNER()
examples = ["Sugar, lnvert syrup, ref1ned pa1m oil, Emu1sifier (INS 322), Acidity regu1ator (330), dextr0se",
            "WHEAT FLOUR SUGAR PALM OIL INVERT SUGAR SYRUP MILK SOLIDS EMULSIFIER 471 COLOUR 150d"]
for text in examples:
    print(text)
    for name, system in (("rules", rules), ("DistilBERT", model)):
        ents = system.predict(text)["entities"]
        print(f"  {name:10}", [(e["text"], e["label"]) for e in ents])
""")
explain("Trained DistilBERT/BERT token classifiers (plus a CRF head and an OCR-noise-augmented variant) on silver data and compared them with the rules and a feature-based CRF using one scorer.",
        "The dictionary is precise on clean text but brittle: OCR errors and missing commas break exact matching. A pretrained Transformer uses context, so it can still recognise 'lnvert syrup' as a sugar.",
        "silver training data", "fine-tuned models + comparison tables",
        "Weak supervision: the model is trained on rule labels and then has to generalise beyond the rules. We show this on noisy text where the rules fail, and the final benchmark is the human gold set.")

md("### A6. Error analysis of the models")
code("""
rows = []
for f in sorted(Path("reports/error_analysis").glob("*_errors_noisy_test_0.05.csv")):
    e = pd.read_csv(f)
    rows.append({"system": f.name.split("_errors")[0], **e["error_type"].value_counts().to_dict(), "total": len(e)})
display(pd.DataFrame(rows).fillna(0).set_index("system"))
errors = pd.read_csv(f"reports/error_analysis/{best}_errors_noisy_test_0.05.csv")
display(errors.sample(min(12, len(errors)), random_state=0)[["text", "expected_entity", "expected_label", "predicted_entity", "predicted_label", "error_type", "tags"]])
""")

md("### A7. Exported Hugging Face model")
code("""
print(Path("models/ingredient-ner-distilbert/README.md").read_text()[:1500])
from transformers import pipeline
hf = pipeline("token-classification", model="models/ingredient-ner-distilbert", aggregation_strategy="first")
pd.DataFrame(hf("Sugar, glucose syrup, acidity regulator (INS 330), palm oil"))[["word", "entity_group", "score"]]
""")

# ------------------------------------------------------------------ Person 3
md("""
## Part B: Person 3, knowledge base, OCR, entity linking, app

### B1. Knowledge base
Built by `src/kb/build_knowledge_base.py` from the Open Food Facts taxonomies (ODbL): descriptions only, no health ratings.
""")
code("""
kb = Path("data/knowledge_base")
additives = pd.read_csv(kb / "additives.csv", dtype=str).fillna("")
print({f.name: len(pd.read_csv(f)) for f in sorted(kb.glob("*.csv"))})
display(additives[additives["ins_number"].isin(["330", "211", "102", "471", "955"])][["kb_id", "name", "ner_label", "function_classes_text", "function_description"]])
nutrition = kb / "products_nutrition.csv"
if nutrition.exists():
    display(pd.read_csv(nutrition, dtype={"product_id": str}).dropna(subset=["sugars_100g"]).head(5))
""")

md("### B2. OCR on a real packet photo")
code("""
from src.ocr.ocr_pipeline import read_image
from src.ocr.ocr_cleanup import fix_additive_codes
from src.ocr.section_extraction import extract_ingredients_section
photo = "data/images/packets/8901063162518.jpg"
display(Image(photo, width=420))
ocr = read_image(photo)
print("OCR text (first 600 chars):\\n", ocr["text"][:600])
cleaned = fix_additive_codes(ocr["text"])
section = extract_ingredients_section(cleaned)
print("\\nSection found by:", section["method"]); print(section["text"])
""")
explain("EasyOCR (CRAFT detector + CRNN recogniser) with OpenCV pre-processing, reading-order line grouping, automatic rotation, a targeted fix for OCR errors in additive codes ('I5Od' -> '150d'), and extraction of the ingredients section.",
        "A packet photo contains much more than the ingredient list, and OCR confuses digits with letters exactly where additive codes are.",
        "photo", "ingredient-list text",
        "The code clean-up only touches code-like tokens in brackets or after INS/E, so it cannot invent ingredients.")

md("### B3. Entity linking")
code("""
from src.kb.entity_linking import get_linker
linker = get_linker()
tests = [{"text": "INS 330", "label": "INS_CODE"}, {"text": "503(ii)", "label": "INS_CODE"}, {"text": "soy lecithin", "label": "ADDITIVE"},
         {"text": "potasium sorbate", "label": "PRESERVATIVE"}, {"text": "Acidity regulator", "label": "FUNCTION_CLASS"},
         {"text": "glucose syrup", "label": "SUGAR"}]
pd.DataFrame([{**t, **{k: v for k, v in linker.link(t).items() if k in ("link_method", "kb_id", "canonical_name", "function")}} for t in tests])
""")

md("### B4. The complete pipeline: photo in, result out")
code("""
from src.app.scanner import IngredientScanner
scanner = IngredientScanner("auto")
result = scanner.scan_image(photo)
print("NER model:", result["ner_model"], "| OCR + analysis time:", result["ocr"]["seconds"], "s")
display(pd.DataFrame(result["entities"])[["text", "label", "canonical_name", "function", "link_method"]])
print(json.dumps(result["summary"], indent=1))
print(result["disclaimer"])
""")

md("### B5. End-to-end evaluation on 60 real photos (held-out test products)")
code("""
e2e = load("reports/ocr_end_to_end.json")
print("Photos:", e2e["photos"]); display(e2e["ocr"])
display(pd.DataFrame({s: {"entities exact F1": v["entities_exact"]["f1"], "entities fuzzy F1": v["entities_fuzzy"]["f1"],
                          "additive numbers P": v["additive_numbers"]["precision"], "additive numbers R": v["additive_numbers"]["recall"],
                          "additive numbers F1": v["additive_numbers"]["f1"]} for s, v in e2e["systems"].items()}).T)
""")
md("""
* **CER** = character error rate of the extracted section against the product's typed ingredient list.
  The reference was typed by Open Food Facts contributors and may differ slightly from the photo.
* **Entity F1** compares (label, text) bags. *Fuzzy* accepts small spelling differences.
* **Additive numbers**: which additives the app identifies (through codes or names), compared with the reference.
""")

md("""
### B6. The app

```bash
streamlit run src/app/streamlit_app.py
```

Tabs: **Camera** (take a picture), **Upload photo**, **Example packets** (the 60 real photos), **Paste text**.
Deployment: `Dockerfile` and `docs/DEPLOYMENT.md` (Hugging Face Spaces recommended).
""")
code("""
for shot in sorted(Path("reports/figures").glob("app_*.png")):
    display(Image(str(shot), width=900))
""")

md("""
## Limitations

* **Silver supervision.** Models learn the rules' policy, including some of their mistakes. Final claims
  need the human gold test set.
* **CPU training.** 8,000 sentences, 1–2 epochs. More data or epochs would likely help.
* **OCR.** Curved, glossy or small print causes errors. The photo reference text is crowd-typed.
  Section extraction fails when "Ingredients" is not printed or not recognised.
* **Missing separators.** When OCR loses all commas (second example in A5), neither the rules nor the
  model split the ingredients correctly.
* **English only.** Labels in other languages or scripts are not handled.
* **No health claims.** The app identifies and explains ingredients; it does not rate products.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/ingredient_scanner_person2_person3.ipynb")
print("written", len(cells), "cells")
