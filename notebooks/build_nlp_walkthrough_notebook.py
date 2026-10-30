"""Build notebooks/nlp_walkthrough.ipynb (run from the project root):  python notebooks/build_nlp_walkthrough_notebook.py

One notebook that walks through every NLP concept of the project, in pipeline order, with short code that
calls the project's real modules, live checks on small samples, and the saved results of the long runs.
"""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# Ingredient Scanner: the NLP behind it

Every NLP concept the project uses, in the order the app uses them. Each section has a one-paragraph
explanation, a small piece of code that calls the project's **real** modules (`src/...`), and the
results: live checks on small samples (seconds) and the saved results of the long training runs.

| # | Section | NLP syllabus |
|---|---|---|
| 1 | Dataset and corpus | corpora, text corpus structure |
| 2 | Preprocessing: normalisation, tokenisation, regular expressions, sub-words | Unit 2, Unit 3 (regex) |
| 3 | Weak supervision and BIO tagging | annotated corpora, chunking |
| 4 | Named entity recognition: rules vs CRF vs DistilBERT | Unit 4 (rule-based, stochastic, ML tagging, NER) |
| 5 | Evaluation: precision, recall, F1, noise robustness, error analysis | evaluation |
| 6 | Spelling correction and edit distance | lexical similarity |
| 7 | OCR quality: character and word error rate (CER, WER) | evaluation |
| 8 | N-gram language model, smoothing and perplexity | Unit 3 (n-grams, smoothing) |
| 9 | Entity linking, synonymy, fuzzy matching | Unit 5 (lexical relations) |
| 10 | Text representations: BoW, TF-IDF, cosine / Jaccard / Euclidean / Manhattan | Unit 2 |
| 11 | Word embeddings: ingredient2vec (word2vec) | Unit 5 (distributional semantics, word2vec) |
| 12 | Base paper and extension: noise-aware training (NAT) | research paper |
| 13 | The ingredient assistant: intents, entities, reference resolution, QA | Units 6-7 |
| 14 | Summary of results | |

Run from top to bottom (about 3-5 minutes on a laptop CPU).
""")

# --------------------------------------------------------------------------------------------- setup
md("## 0. Setup")
code("""
import os, sys, json, re, warnings
from collections import Counter
from pathlib import Path
os.environ["USE_TF"] = "0"          # this laptop's global Python has a broken TensorFlow; transformers must skip it
os.environ["USE_TORCH"] = "1"
warnings.filterwarnings("ignore")

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
pd.set_option("display.max_colwidth", 90)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

def show(path, width=760):
    display(Image(filename=str(ROOT / path), width=width))

def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))

print("project folder:", ROOT)
""")

# --------------------------------------------------------------------------------------------- 1
md("""
## 1. Dataset and corpus

**Open Food Facts**, an open crowd-sourced database (ODbL licence). From 4.5 million products we kept English
ingredient lists from English-speaking markets, removed exact duplicates, and sampled per market:
**22,355 products**. Each product's ingredient list is one document of our corpus.
""")
code("""
products = pd.read_csv("data/processed/products.csv", dtype={"product_id": str})
funnel = pd.DataFrame({"step": ["Full Open Food Facts export", "English-speaking markets", "has an ingredient list",
                                "list written in English", "unique ingredient lists", "project dataset (sampled per market)"],
                       "products": [4_535_553, 1_441_888, 531_903, 509_294, 405_812, len(products)]})
display(funnel)
display(products["country_group"].value_counts().rename("products").to_frame())
print(products["n_chars"].describe().round(1).to_string())
products[["product_name", "ingredients_text_raw"]].dropna().sample(3, random_state=1)
""")
code("""
show("reports/figures/02_ingredient_length_distribution.png", 620)
""")

# --------------------------------------------------------------------------------------------- 2
md("""
## 2. Preprocessing: normalisation, tokenisation, regular expressions, sub-words

* **Normalisation**: unify Unicode, spacing and punctuation so the same text always looks the same.
* **Tokenisation with character offsets**: every token remembers where it is in the text, so labels can
  be mapped back to the exact characters (needed for highlighting and for BIO tags).
* **Regular expressions**: additive codes are written many ways (`INS 330`, `E-330`, `(330)`, `503(ii)`);
  one set of patterns maps them all to a canonical number.
* **Sub-word tokenisation (WordPiece)**: DistilBERT splits rare or misspelled words into known pieces,
  so it can still process words it has never seen.
""")
code("""
from src.preprocessing.pipeline import process_ingredient_text
from src.preprocessing.additive_codes import normalize_code

p = process_ingredient_text("INGREDIENTS:  Sugar,  Palm Oil , Acidity Regulator (INS 330), Emulsifier E-471, "
                            "Raising agent 503(ii), Hazelnuts (13%)")
print("normalised text:", p["text"])
display(pd.DataFrame({"token": p["tokens"], "start": [s for s, _ in p["token_offsets"]],
                      "end": [e for _, e in p["token_offsets"]]}).head(14).T)
print("additive codes found:", [(c["text"], c["number"]) for c in p["additive_codes"]])
print("percentages found:   ", [(x["text"], x["value"]) for x in p["percentages"]])
""")
code("""
for code_text in ["INS 330", "E-330", "E330", "330", "503(ii)", "INS 503 (ii)", "E150d", "Hazelnuts"]:
    print(f"{code_text:14} -> {normalize_code(code_text)}")
""")
code("""
from transformers import AutoTokenizer
wordpiece = AutoTokenizer.from_pretrained("models/ingredient-ner-distilbert")
for word in ["sugar", "sucralose", "maltodextrin", "dextr0se", "carrageenan"]:
    print(f"{word:14} -> {wordpiece.tokenize(word)}")
""")

# --------------------------------------------------------------------------------------------- 3
md("""
## 3. Weak supervision and BIO tagging

No labelled dataset existed, so the training labels were made **automatically** (*weak supervision*):
curated dictionaries (sugars, fats, sweeteners, flavourings, additive names and codes) plus rules
(e.g. a segment whose last word is *oil* is a FAT). These are **silver** labels: not checked by humans.

Entities are encoded with **BIO tags** (a form of chunking): `B-` = first word of an entity, `I-` = inside,
`O` = outside. 10 entity classes -> 21 tags.
""")
code("""
from src.labeling.weak_labeler import label_text
from src.labeling.bio import entities_to_bio

lab = label_text("Sugar, glucose syrup, refined palm oil, acidity regulator (INS 330), natural flavourings")
display(pd.DataFrame(lab["entities"])[["text", "label", "rule"]])
bio = entities_to_bio(lab["token_offsets"], lab["entities"])
display(pd.DataFrame({"token": lab["tokens"], "BIO tag": bio}).T)
""")
code("""
from src.ner.data import load_split, noisy_copy, tags_of
split_sizes = {s: len(load_split("silver", s)) for s in ("train", "validation", "test")}
print("silver corpus split (products with the same name/brand are kept in one split):", split_sizes)
train = load_split("silver", "train")
counts = Counter(e["label"] for r in train for e in r["entities"])
print(f"training entities: {sum(counts.values()):,}")
ax = pd.Series(counts).sort_values().plot.barh(figsize=(6, 3.4), color="#2a78d6")
ax.set_title("Silver entities per class (training split)"); ax.set_xlabel("entities"); plt.show()
""")

# --------------------------------------------------------------------------------------------- 4
md("""
## 4. Named entity recognition: rules vs CRF vs DistilBERT

The core NLP task: label every ingredient with one of 10 classes. Three families of tagger:

| System | Type | Idea |
|---|---|---|
| Dictionary rules | rule-based | exact dictionary and head-word matches (these also made the silver labels) |
| CRF | statistical sequence model | word-shape, affix and context features; learns tag transitions |
| DistilBERT | transformer (deep learning) | pretrained language model fine-tuned on the silver labels, with OCR-noise augmentation |

Below, all three tag the same OCR-damaged list. The rules miss words with OCR slips (`dextr0se`, `oii`);
the noise-trained DistilBERT recovers most of them.
""")
code("""
from src.baseline.dictionary_ner import DictionaryNER
from src.ner.crf_baseline import CRFTagger
from src.ner.transformer_ner import TransformerTagger

dictionary_ner = DictionaryNER()
crf = CRFTagger()
distilbert = TransformerTagger("models/ingredient-ner-distilbert")

def dictionary_tags(records):
    return [entities_to_bio(r["token_offsets"], dictionary_ner.predict_record(r)) for r in records]

ocr_like = "sugar, dextr0se, palm oii, acidity regulator (INS 33O), maltodexirin, colour (caramel E150d)"
p = process_ingredient_text(ocr_like)
display(pd.DataFrame({"token": p["tokens"], "rules": dictionary_tags([p])[0],
                      "CRF": crf.predict_tags(p["tokens"]), "DistilBERT": distilbert.predict_tags(p["tokens"])}).T)
""")

# --------------------------------------------------------------------------------------------- 5
md("""
## 5. Evaluation: precision, recall, F1, robustness to OCR noise, error analysis

**Strict entity-level micro F1**: an entity counts as correct only if its span *and* class match exactly.
Because photos produce OCR errors, every system is also tested on the same test lists with synthetic
OCR-style character confusions (`l`<->`1`, `O`<->`0`, ...) at 5% and 10% of characters.

**Live check** on 300 held-out test products (runs in under a minute):
""")
code("""
from src.evaluation.metrics import evaluate_tag_sequences

test = load_split("silver", "test", limit=300, seed=7)
eval_sets = {"clean": test, "5% OCR noise": noisy_copy(test, 0.05, seed=1), "10% OCR noise": noisy_copy(test, 0.10, seed=2)}
systems = {"dictionary rules": dictionary_tags, "CRF": crf.predict_records, "DistilBERT + noise aug.": distilbert.predict_records}
rows = []
for name, predict in systems.items():
    for set_name, records in eval_sets.items():
        micro = evaluate_tag_sequences([tags_of(r) for r in records], predict(records))["micro"]
        rows.append({"system": name, "test set": set_name, "precision": micro["precision"],
                     "recall": micro["recall"], "F1": micro["f1"]})
live = pd.DataFrame(rows)
display(live.pivot(index="system", columns="test set", values="F1")[["clean", "5% OCR noise", "10% OCR noise"]].round(3))
live.pivot(index="system", columns="test set", values="F1")[["clean", "5% OCR noise", "10% OCR noise"]].plot.bar(
    figsize=(7, 3.4), rot=0, ylim=(0.5, 1.02), title="Entity F1 (live, 300 test products)"); plt.show()
""")
md("""
On clean text the rules score 1.0 **by construction** (they made the reference labels). The honest
comparison is on noisy text, where the noise-trained transformer is clearly the most robust.

**Full saved results** (1,000 test products, all six systems, from `python -m src.ner.evaluate_models`):
""")
code("""
summary = pd.DataFrame(load("reports/model_comparison.json")["summary_f1"]).T
display(summary.round(3))
show("reports/figures/12_model_comparison.png")
""")
code("""
details = load("reports/model_comparison.json")["details"]["distilbert_aug"]["noisy_test_0.05"]["per_class"]
display(pd.DataFrame(details).T[["precision", "recall", "f1", "support"]].round(3))
show("reports/figures/11_noise_robustness.png", 560)
""")
md("**Error analysis** of the augmented DistilBERT on the 5% noise set: which kinds of mistakes does it make?")
code("""
errors = pd.read_csv("reports/error_analysis/distilbert_aug_errors_noisy_test_0.05.csv")
display(errors["error_type"].value_counts().rename("errors").to_frame())
errors[["expected_entity", "expected_label", "predicted_entity", "predicted_label", "error_type"]].dropna(subset=["expected_entity"]).sample(6, random_state=3)
""")

# --------------------------------------------------------------------------------------------- 6
md("""
## 6. Spelling correction and edit distance

OCR slips are repaired with **SymSpell** and a 6,400-word food dictionary built from the 22,000 real lists.
**Levenshtein (edit) distance** = the number of single-character insertions, deletions and substitutions
between two strings; it drives correction, fuzzy matching and the OCR error rates below.
""")
code("""
from src.ocr.text_correction import correct_ocr_text
from rapidfuzz.distance import Levenshtein

raw = "flavcurenhancers, maltodexrin, slarch, dextr0se, Emu1sifier (INS 322), regu1ator (330)"
print("OCR text :", raw)
print("corrected:", correct_ocr_text(raw))
for a, b in [("maltodexirin", "maltodextrin"), ("dextr0se", "dextrose"), ("sugar", "salt")]:
    print(f"edit distance {a!r} -> {b!r}: {Levenshtein.distance(a, b)}  (similarity {Levenshtein.normalized_similarity(a, b):.2f})")
""")

# --------------------------------------------------------------------------------------------- 7
md("""
## 7. OCR quality: character and word error rate (CER, WER)

For the 60 real packet photos we know the product's typed ingredient list (the reference). We compare it
with the ingredient section the app extracted from the photo:

* **CER** = character edit distance / reference length
* **WER** = **word** edit distance (substituted + deleted + inserted words) / number of reference words

Version 1 of the app used EasyOCR; version 2 uses RapidOCR (PP-OCRv6) with enlargement, rotation and
spelling correction. (WER can exceed 1 when the extracted section contains extra text.)
""")
code("""
from src.ocr.section_extraction import extract_ingredients_section

reference = {r["id"]: r["text"] for r in load_split("silver", "test")}
words = lambda t: re.findall(r"[a-z0-9]+", t.lower())

def error_rates(cache_path):
    rows = []
    for line in (ROOT / cache_path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["product_id"] not in reference:
            continue
        section = extract_ingredients_section(r["corrected_text"])["text"]
        ref = reference[r["product_id"]]
        rows.append({"product": r["product_id"],
                     "CER": Levenshtein.normalized_distance(" ".join(words(ref)), " ".join(words(section))),
                     "WER": Levenshtein.distance(words(ref), words(section)) / max(1, len(words(ref)))})
    return pd.DataFrame(rows)

v1 = error_rates("reports/ocr_easyocr_v1/ocr_results.jsonl")
v2 = error_rates("data/images/ocr_results.jsonl")
table = pd.DataFrame({"photos": [len(v1), len(v2)], "median CER": [v1.CER.median(), v2.CER.median()],
                      "median WER": [v1.WER.median(), v2.WER.median()],
                      "photos with WER < 0.3": [(v1.WER < 0.3).sum(), (v2.WER < 0.3).sum()]},
                     index=["v1: EasyOCR", "v2: RapidOCR + correction"])
display(table.round(3))
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.boxplot([v1.WER.clip(upper=1.5), v2.WER.clip(upper=1.5)], labels=["v1: EasyOCR", "v2: RapidOCR + corr."])
ax.set_ylabel("word error rate (clipped at 1.5)"); ax.set_title("WER of the extracted ingredient section, 60 photos"); plt.show()
""")
md("**One live scan** of a real photo (OCR -> clean-up -> section -> hybrid NER -> knowledge base):")
code("""
from src.app.scanner import IngredientScanner
scanner = IngredientScanner()
result = scanner.scan_image("data/images/unseen/8000500225615.jpg")
print({k: result["ocr"][k] for k in ("engine", "rotation", "scale", "n_lines", "mean_confidence", "section_method", "seconds")})
print("\\ningredient section read:\\n", result["text"][:300])
display(pd.DataFrame(result["entities"])[["text", "label", "source", "canonical_name", "ins_number"]].head(12))
""")

# --------------------------------------------------------------------------------------------- 8
md("""
## 8. N-gram language model, smoothing and perplexity

A **bigram model** estimates P(word | previous word) from counts in the training lists; a **trigram**
model uses the two previous words. Unseen word pairs would get probability 0, so the counts are
**smoothed**: add-k (Laplace when k = 1) or **interpolation** (mix trigram, bigram and unigram estimates).

**Perplexity** = 2^(average negative log2 probability per word): how "surprised" the model is by new text.
Lower = more predictable. We compare the models on held-out test lists, then show how much OCR errors
make text less predictable.
""")
code("""
import math

def lm_words(text):
    return ["<s>", "<s>"] + re.findall(r"[a-z]+|\\d+|[,;:()\\[\\]]", text.lower()) + ["</s>"]

train_texts = [r["text"] for r in train]
validation_texts = [r["text"] for r in load_split("silver", "validation", limit=1000, seed=1)]
test_texts = [r["text"] for r in load_split("silver", "test", limit=1000, seed=1)]

unigram_counts = Counter(w for t in train_texts for w in lm_words(t)[2:])
VOCAB = {w for w, c in unigram_counts.items() if c >= 2} | {"<unk>", "</s>"}
def tokens(text):
    return [w if w in VOCAB or w == "<s>" else "<unk>" for w in lm_words(text)]

class NGramLM:
    def __init__(self, texts):
        self.uni, self.bi, self.tri = Counter(), Counter(), Counter()
        self.ctx1, self.ctx2 = Counter(), Counter()
        for t in texts:
            w = tokens(t)
            for i in range(2, len(w)):
                self.uni[w[i]] += 1
                self.bi[(w[i-1], w[i])] += 1; self.ctx1[w[i-1]] += 1
                self.tri[(w[i-2], w[i-1], w[i])] += 1; self.ctx2[(w[i-2], w[i-1])] += 1
        self.total, self.V = sum(self.uni.values()), len(VOCAB)
    def p_unigram(self, w):            return (self.uni[w] + 1) / (self.total + self.V)
    def p_bigram(self, a, w, k):       return (self.bi[(a, w)] + k) / (self.ctx1[a] + k * self.V)
    def p_trigram(self, a, b, w, k):   return (self.tri[(a, b, w)] + k) / (self.ctx2[(a, b)] + k * self.V)
    def p_interp(self, a, b, w, l):
        p3 = self.tri[(a, b, w)] / self.ctx2[(a, b)] if self.ctx2[(a, b)] else 0
        p2 = self.bi[(b, w)] / self.ctx1[b] if self.ctx1[b] else 0
        return l[0] * p3 + l[1] * p2 + l[2] * self.p_unigram(w)

def perplexity(prob, texts):
    log_sum, n = 0.0, 0
    for t in texts:
        w = tokens(t)
        for i in range(2, len(w)):
            log_sum += -math.log2(prob(w[i-2], w[i-1], w[i])); n += 1
    return 2 ** (log_sum / n)

lm = NGramLM(train_texts)
print(f"vocabulary {lm.V:,} word types, {lm.total:,} training tokens")
""")
code("""
# interpolation weights: choose the best on the VALIDATION lists (never on the test lists)
grid = [(a, b, round(1 - a - b, 2)) for a in (0.2, 0.4, 0.6) for b in (0.2, 0.3, 0.4) if a + b < 1]
best_l = min(grid, key=lambda l: perplexity(lambda a, b, w: lm.p_interp(a, b, w, l), validation_texts))
print("interpolation weights (trigram, bigram, unigram) chosen on validation:", best_l)

models = {
    "unigram (add-1)":            lambda a, b, w: lm.p_unigram(w),
    "bigram, Laplace (add-1)":    lambda a, b, w: lm.p_bigram(b, w, 1.0),
    "bigram, add-k (k=0.01)":     lambda a, b, w: lm.p_bigram(b, w, 0.01),
    "trigram, add-k (k=0.01)":    lambda a, b, w: lm.p_trigram(a, b, w, 0.01),
    "trigram, interpolated":      lambda a, b, w: lm.p_interp(a, b, w, best_l),
}
ppl = pd.Series({name: perplexity(prob, test_texts) for name, prob in models.items()}, name="test perplexity")
display(ppl.round(1).to_frame())
ppl.plot.barh(figsize=(6, 2.8), color="#1baf7a", title="Perplexity on 1,000 held-out lists (lower is better)"); plt.show()
""")
md("""
Laplace smoothing gives too much probability to unseen pairs, so the bigram model with add-1 does
*worse* than with a small k; interpolation does best. **Next-word prediction** with the best model:
""")
code("""
best = models["trigram, interpolated"]
for context in [("glucose",), ("palm",), ("acidity",), ("skimmed", "milk")]:
    a, b = (["<s>"] + list(context))[-2:]
    ranked = sorted(VOCAB - {"<unk>"}, key=lambda w: -best(a, b, w))[:5]
    print(f"after {' '.join(context)!r:22} -> {ranked}")
""")
md("**How much do OCR errors disrupt the text?** Perplexity of the same test lists, clean vs with OCR noise, and of the sections read from real photos:")
code("""
noisy5 = [r["text"] for r in noisy_copy(load_split("silver", "test", limit=1000, seed=1), 0.05, seed=1)]
noisy10 = [r["text"] for r in noisy_copy(load_split("silver", "test", limit=1000, seed=1), 0.10, seed=2)]
photos = [extract_ingredients_section(json.loads(l)["corrected_text"])["text"]
          for l in (ROOT / "data/images/ocr_results.jsonl").read_text(encoding="utf-8").splitlines()]
photo_refs = [reference[json.loads(l)["product_id"]] for l in (ROOT / "data/images/ocr_results.jsonl").read_text(encoding="utf-8").splitlines()]
display(pd.Series({"test lists, clean": perplexity(best, test_texts), "same lists, 5% OCR noise": perplexity(best, noisy5),
                   "same lists, 10% OCR noise": perplexity(best, noisy10),
                   "60 photos: typed reference": perplexity(best, photo_refs),
                   "60 photos: section read by the app": perplexity(best, photos)},
                  name="perplexity").round(1).to_frame())
""")

# --------------------------------------------------------------------------------------------- 9
md("""
## 9. Entity linking, synonymy and fuzzy matching

Each recognised entity is **linked** to the knowledge base (731 additives from the Open Food Facts
taxonomy, function classes, ingredient terms). Methods, tried in order: INS/E number -> exact name or
**synonym** (E330 = INS 330 = citric acid) -> official additive name -> **fuzzy** name (Levenshtein
similarity >= 88, for OCR slips) -> category description only.
""")
code("""
from src.kb.entity_linking import get_linker
linker = get_linker()
examples = [{"text": "INS 330", "label": "INS_CODE"}, {"text": "E150d", "label": "INS_CODE"},
            {"text": "soy lecithin", "label": "ADDITIVE"}, {"text": "lecithln", "label": "ADDITIVE"},
            {"text": "sodium benzoate", "label": "PRESERVATIVE"}, {"text": "Yellow 5", "label": "COLOUR"}]
pd.DataFrame([{"entity": e["text"], **{k: v for k, v in linker.link(e).items()
               if k in ("canonical_name", "ins_number", "function", "link_method", "link_score")}} for e in examples])
""")

# --------------------------------------------------------------------------------------------- 10
md("""
## 10. Text representations and similarity measures

When a photo shows no nutrition table, the app estimates nutrition from the **most similar product** in
the database. Which representation finds the best match?

* **Bag of Words (BoW)**: a list = its word counts
* **TF-IDF**: counts weighted by rarity (common words like *salt* count less)
* **Jaccard**: shared words / all words; **cosine**: angle between vectors;
  **Euclidean** and **Manhattan**: distances between vectors

A small example first:
""")
code("""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

docs = {"chocolate spread": "sugar, palm oil, hazelnuts, cocoa, skimmed milk powder, lecithin, vanillin",
        "hazelnut cream":   "sugar, hazelnuts, vegetable oil, cocoa powder, milk powder, emulsifier lecithin",
        "potato chips":     "potato, palm oil, salt, maltodextrin, flavour enhancer, citric acid"}
bow = CountVectorizer().fit_transform(docs.values())
tfidf = TfidfVectorizer().fit_transform(docs.values())
sets = [set(re.findall(r"[a-z]+", d)) for d in docs.values()]
names = list(docs)
print("Jaccard:"); display(pd.DataFrame([[len(a & b) / len(a | b) for b in sets] for a in sets], names, names).round(2))
print("Cosine on TF-IDF:"); display(pd.DataFrame(cosine_similarity(tfidf), names, names).round(2))
print("Euclidean on BoW (distance, lower = closer):"); display(pd.DataFrame(euclidean_distances(bow), names, names).round(2))
print("Manhattan on BoW (distance, lower = closer):"); display(pd.DataFrame(manhattan_distances(bow), names, names).round(0))
""")
md("""
**Evaluation** (`python -m src.evaluation.product_similarity`): for 1,320 held-out test products, the most
similar product among 6,889 database products predicts the nutrition; lower error = better representation.
""")
code("""
sim = load("reports/product_similarity.json")
table = pd.DataFrame({name: {"sugars MAE (g)": r["top1"]["sugars_g"]["mae"], "fat MAE (g)": r["top1"]["fat_g"]["mae"],
                             "salt MAE (g)": r["top1"]["salt_g"]["mae"], "energy MAE (kcal)": r["top1"]["energy_kcal"]["mae"]}
                      for name, r in sim.items()}).T
display(table)
table["energy MAE (kcal)"].sort_values().plot.barh(figsize=(7, 3.6), color="#2a78d6",
    title="Energy prediction error by representation (lower is better)"); plt.show()
""")

# --------------------------------------------------------------------------------------------- 11
md("""
## 11. Word embeddings: ingredient2vec (word2vec)

**Distributional semantics**: words that occur in similar contexts have similar meanings. We trained
**word2vec (CBOW)** on the 14,000 training lists, with each ingredient as one token (`glucose_syrup`),
so ingredients used in the same way end up close together, even with no letters in common.
""")
code("""
from src.embeddings.ingredient2vec import load_model, most_similar
w2v = load_model()
print(f"{len(w2v.wv):,} ingredient vectors of {w2v.vector_size} dimensions")
for term in ["sucralose", "citric_acid", "palm_oil", "dextrose", "soya_lecithin"]:
    print(f"{term:15} -> {[t for t, _ in most_similar(term, 5, w2v)]}")
""")
md("""
**Evaluation: do the vectors know an ingredient's category?** 866 terms whose category the rules know,
classified by k-nearest neighbours (5-fold cross-validation). The vectors never saw the labels.
""")
code("""
emb = load("reports/embeddings/ingredient2vec.json")
c = emb["classification"]
display(pd.DataFrame({"accuracy": [c["majority_class"]["accuracy"], c["spelling_char3gram_knn"]["accuracy"],
                                   c["ingredient2vec_knn"]["accuracy"], c["spelling_plus_ingredient2vec_knn"]["accuracy"]],
                      "macro F1": [None, c["spelling_char3gram_knn"]["macro_f1"], c["ingredient2vec_knn"]["macro_f1"],
                                   c["spelling_plus_ingredient2vec_knn"]["macro_f1"]]},
                     index=["majority class", "spelling (char 3-grams)", "ingredient2vec (context)", "spelling + context"]))
print("Configurations compared (window x architecture):")
display(pd.DataFrame(emb["grid"]))
show("reports/figures/14_ingredient2vec_tsne.png", 640)
""")

md("""
### 11b. Using the embeddings: sub-groups for "Other ingredients"

The NER schema has one generic INGREDIENT class. Those ingredients are sorted afterwards into sub-groups
(protein, fibre, probiotics, enzymes, grains, dairy, spices ...): first by ordered **name rules**, and for
names no rule covers by **k-nearest neighbours in the ingredient2vec space** (similarity-weighted vote of
the closest rule-labelled ingredients; only used when the neighbours agree).
""")
code("""
from src.kb.ingredient_subgroups import INFO, subgroup_of
for name in ["Chicory Root Fiber", "Probiotic Bacillus Coagulans", "Bromelain & Papain", "Whey protein concentrate",
             "Hydrolysed Groundnut Protein", "Gram Meal", "kasuri methi", "Sultanas"]:
    group, method = subgroup_of(name)
    print(f"{name:30} -> {INFO[group]['icon']} {INFO[group]['title']:28} ({method or 'no group'})")
sub = load("reports/ingredient_subgroups.json")
print(f"\\nembedding fallback, tested on {sub['rule_labelled_terms']:,} held-out rule-labelled terms: answers "
      f"{sub['coverage']:.0%} of them with {sub['accuracy_when_answered']:.1%} accuracy "
      f"(majority baseline {sub['majority_baseline']['accuracy']:.1%})")
display(pd.DataFrame(sub["grid"], columns=["min similarity", "min share of votes", "answered", "accuracy"]))
""")

# --------------------------------------------------------------------------------------------- 12
md("""
## 12. Base paper and extension: noise-aware training (NAT)

Base paper: Namysl et al., *NAT: Noise-Aware Training for Robust Neural Sequence Labeling* (ACL 2020).
Our extension: (1) an OCR noise model **learned** from our real photos instead of hand-picked confusions,
(2) NAT's **stability training** loss: CE(clean) + alpha * KL(p_clean || p_noisy). Four DistilBERT models,
same budget (2,000 sentences, 2 epochs, seed 42).
""")
code("""
noise = load("data/processed/learned_ocr_noise.json")
print(f"learned from {noise['photos_used']} photos, {noise['characters']:,} letters/digits:")
print({k: f"{v:.2%}" for k, v in noise["global"].items()}, "per character")
print("most common substitutions:", noise["top_substitutions"][:8])
nat = pd.DataFrame(load("reports/nat_extension/model_comparison.json")["summary_f1"]).T
display(nat.drop(columns=["sentences_per_second"]).round(3))
show("paper/figures/fig_nat_comparison.png", 520)
""")

# --------------------------------------------------------------------------------------------- 13
md("""
## 13. The ingredient assistant: intent classification, entities, reference resolution, QA

A **retrieval-based, closed-domain question-answering system** (not a generative chatbot): every answer
comes from the knowledge base or the scan, with its source.

1. **Intent classification** (text classification): TF-IDF of words and character n-grams + logistic
   regression, compared with a keyword-rules baseline
2. **Entity extraction**: which ingredient / code is the question about (codes, names, fuzzy names)
3. **Reference resolution**: "is **it** vegan?" -> the ingredient discussed before
4. **Out-of-domain detection**: off-topic questions are declined
5. **Template-based answers** with sources; medical questions are declined
""")
code("""
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from src.assistant.intents import model_intent, rule_intent
from src.utils.io import read_jsonl

questions = read_jsonl("data/assistant/test_questions.jsonl") + read_jsonl("data/assistant/test_questions_fresh.jsonl")
gold = [q["intent"] for q in questions]
pred_model = [model_intent(q["question"])[0] for q in questions]
pred_rules = [rule_intent(q["question"]) for q in questions]
print(f"{len(questions)} held-out questions: intent accuracy rules {np.mean(np.array(pred_rules) == gold):.1%}, "
      f"model {np.mean(np.array(pred_model) == gold):.1%}")
print(classification_report(gold, pred_model, zero_division=0))
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_predictions(gold, pred_model, ax=ax, xticks_rotation=60, colorbar=False, cmap="Blues")
ax.set_title("Intent classifier: confusion matrix (held-out questions)"); plt.tight_layout(); plt.show()
""")
md("**A conversation** about a scanned product (Nutella sample):")
code("""
from src.assistant.assistant import IngredientAssistant
nutella = scanner.scan_text("Ingredients: Sugar, Palm Oil, Hazelnuts (13%), Skimmed Milk Powder (8.7%), "
                            "Fat-Reduced Cocoa (7.4%), Emulsifier: Lecithins (Soya), Vanillin.")
bot = IngredientAssistant(nutella, body_weight_kg=60)
history = []
for question in ["What is E322?", "Is it vegan?", "Which allergens does it contain?", "Is titanium dioxide banned?",
                 "My son has diabetes, can he eat this?", "What is the speed of light?"]:
    reply = bot.answer(question, history); history.append(reply)
    ref = f"  [“it” = {reply['resolved_reference']}]" if reply["resolved_reference"] else ""
    print(f"Q: {question}   -> intent: {reply['intent']}{ref}")
    print("A:", re.sub(r"\\*\\*", "", reply["text"])[:260], "\\n")
""")
code("""
ev = load("reports/assistant_evaluation.json")
fr, dev = ev["fresh"], ev["development"]
pd.DataFrame({"development set (64)": [dev["intent_accuracy"]["rules"], dev["intent_accuracy"]["model_plus_safety_net"],
                                        dev["mention_accuracy"], dev["answer_correct"], dev["refusal"]["recall"]],
              "fresh set (24, run once)": [fr["intent_accuracy"]["rules"], fr["intent_accuracy"]["model_plus_safety_net"],
                                           fr["mention_accuracy"], fr["answer_correct"], fr["refusal"]["recall"]]},
             index=["intent accuracy: rules", "intent accuracy: model", "right ingredient found",
                    "answer contains every required fact", "medical questions declined"]).style.format("{:.1%}")
""")

# --------------------------------------------------------------------------------------------- 14
md("""
## 14. Summary of results

| Component | Method | Result |
|---|---|---|
| NER, clean text | DistilBERT + noise augmentation | F1 0.94 (rules 1.0 by construction) |
| NER, 10% OCR noise | DistilBERT + noise augmentation | **F1 0.90** vs rules 0.73, CRF 0.80, BERT 0.77 |
| OCR, 60 real photos | RapidOCR + spelling correction | median CER 0.62 -> 0.22, median **WER 0.89 -> 0.30** |
| Additives from photos | full pipeline | F1 0.64 -> **0.87** |
| Product similarity | TF-IDF + ingredient2vec | energy error 91.5 -> **79.7 kcal** (vs TF-IDF alone) |
| Ingredient categories from embeddings | spelling + word2vec, kNN | **93.4%** accuracy (spelling alone 90.5%) |
| Sub-groups of "Other ingredients" | name rules + word2vec kNN fallback | fallback **82.7%** accurate on the 51% it answers (baseline 20.8%) |
| Language model | interpolated trigram | test perplexity **21.0** (Laplace bigram 109.0); OCR-read text 66.9 vs typed 24.7 |
| NAT extension | stability training + learned noise | best DistilBERT on every noisy test set |
| Assistant | intent model + knowledge base | 91.7% correct answers on the fresh set, 100% medical refusals |

**Limitations**: the NER labels are silver (made by rules; a 400-product gold set is prepared but not
annotated), English only, and OCR remains the main source of errors on real photos.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/nlp_walkthrough.ipynb")
print("written", len(cells), "cells")
