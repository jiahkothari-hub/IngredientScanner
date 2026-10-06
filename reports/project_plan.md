# Person 1 (DATA + NLP) — Plan and design decisions

This file records *why* the data pipeline is built the way it is. The detailed annotation rules
will live in `reports/entity_schema.md` (Stage 3); this file is the proposal it is based on.

## 1. Architecture

```
IngredientScanner/
├── configs/data_config.yaml      every tunable number (quotas, thresholds, seed, URLs) in one place
├── data/
│   ├── raw/                      untouched downloads (git-ignored; manifest + taxonomy committed)
│   ├── interim/                  re-creatable intermediate files (git-ignored)
│   ├── processed/                project dataset + reference tables (committed: the frozen snapshot)
│   ├── annotations/              gold annotation batches (human-verified only)
│   └── splits/                   train / validation / test files for Person 2
├── src/
│   ├── data/                     STAGE 1: download, filter, taxonomy, EDA
│   ├── preprocessing/            STAGE 2: process_ingredient_text() — shared with Person 3 (OCR)
│   ├── labeling/                 STAGE 3: dictionaries, regex rules, weak labeller -> silver data
│   ├── annotation/               STAGE 4: gold sampling, Doccano export/import, agreement
│   ├── baseline/                 STAGE 6: dictionary/rule NER = the baseline Person 2 must beat
│   ├── evaluation/               STAGE 6-7: entity-level P/R/F1, error analysis
│   └── utils/                    config/paths, JSONL I/O, validation checks
├── reports/                      figures, statistics, schema, metrics, error analysis
└── tests/                        pytest unit tests for every stage
```

Integration contracts (the only things other people depend on):

| Consumer | What they use | Where |
|---|---|---|
| Person 2 (NER models) | `train/validation/test.jsonl` with `tokens`, `ner_tags`, `label2id.json` | `data/splits/` |
| Person 2 | entity-level scorer, so all models are scored identically | `src/evaluation/` |
| Person 3 (OCR + app) | `process_ingredient_text(text)` | `src/preprocessing/` |
| Person 3 (knowledge base) | additive table: number → name → functional classes | `data/processed/additives_reference.csv` |

## 2. Dataset / source

**Open Food Facts full CSV export** (`en.openfoodfacts.org.products.csv.gz`, ODbL licence),
downloaded from the official URL with the official AWS S3 mirror as fallback.

| Property | Value (snapshot 2026-10-01) |
|---|---|
| Compressed size | 1.28 GB (13 GB uncompressed, ~200 columns) |
| Products | 4,535,553 |
| `ingredients_text` missing (incl. literal "undefined") | 72.4% |
| Columns we read | 10 (`code`, `product_name`, `brands`, `countries_en`, `categories_en`, `main_category_en`, `ingredients_text`, `ingredients_tags`, `additives_n`, `additives_tags`) |
| Products sold in India | 21,188 (only ~4,400 with a usable English ingredient list) |

Why not the API: it is rate-limited and meant for apps. Why not keep everything: we need a dataset we
can inspect, annotate and explain. The project dataset is **22,355 products**: all usable Indian
products (INS-style labels) plus seeded samples from UK/Ireland (E-numbers), North America (names
such as "Red 40") and other English-speaking markets.

Second source: the **Open Food Facts additive taxonomy** (pinned GitHub commit) → 731 additive numbers
with English names, synonyms and functional classes. INS and E numbers share one numbering system
(INS 330 = E330 = citric acid), so one table serves both notations.

## 3. Entity schema (final: 10 labels)

The original suggestion mixed two questions: *what is this mention?* and *what job does this additive
do?* The second question causes overlaps, because citric acid is an antioxidant, a sequestrant and an
acidity regulator at once. The team decided to keep **PRESERVATIVE** and **COLOUR** as their own NER labels
(they matter most to consumers and are lexically distinctive), and to handle every other function through
the `FUNCTION_CLASS` span plus linking.

| Label | Examples |
|---|---|
| `SUGAR` | sugar, glucose syrup, dextrose, honey, jaggery |
| `SWEETENER` | sucralose, aspartame, sorbitol, steviol glycosides |
| `FAT` | palm oil, refined palmolein, ghee, cocoa butter |
| `PRESERVATIVE` | sodium benzoate, potassium sorbate, sulphur dioxide |
| `COLOUR` | tartrazine, caramel colour, Red 40, annatto |
| `ADDITIVE` | citric acid, soy lecithin, xanthan gum, mono- and diglycerides |
| `INS_CODE` | INS 330, E330, (471), 503(ii) |
| `FUNCTION_CLASS` | acidity regulator, emulsifier, preservative, colour |
| `FLAVOURING` | natural flavour, nature-identical flavouring substances |
| `INGREDIENT` | wheat flour, milk solids, salt, water |

**How PRESERVATIVE / COLOUR / ADDITIVE are separated without guesswork:** a named substance takes its label
from its INS number. The INS system groups numbers by main function (100–199 colours, 200–299
preservatives, …). The exceptions are listed in `configs/lexicons/additive_label_rules.yaml`, e.g. acids
260–297 are ADDITIVE and calcium carbonate 170 is ADDITIVE. The word "preservative" or "colour" on a
label is `FUNCTION_CLASS`, never PRESERVATIVE/COLOUR.

Overlap priority: `INS_CODE` > `SWEETENER` > `SUGAR` > `FAT` > `FLAVOURING` >
`PRESERVATIVE`/`COLOUR`/`ADDITIVE` (by number) > `FUNCTION_CLASS` > `INGREDIENT`.
Full rules: `reports/entity_schema.md`.

Three layers, kept separate on purpose:

| Layer | Example | Who |
|---|---|---|
| Identification | "INS 330" is an `INS_CODE` | NER (Person 1 baseline, Person 2 models) |
| Interpretation | INS 330 = citric acid, declared as "acidity regulator" | `src/labeling/interpret.py`, Person 3's knowledge base |
| Health claim | "this is bad for you" | **out of scope**: the system never makes one |

## 4. Annotation strategy

1. **Silver data (automatic).** Dictionaries + regex label all 22k products. Every entity records
   which rule produced it. Silver labels are a training resource, never treated as truth.
2. **Gold data (human-verified).** ~400 products sampled with stratification (market group, length,
   with/without codes, multiple sugars/fats, noisy/OCR-like text). Annotators correct silver
   pre-annotations in **Doccano** (free, local, span labelling, JSONL import/export). Every record
   stores annotator name and status; only `status = verified` records count as gold.
3. **Consistency.** All three team members double-annotate the same ~40 products first; we compute
   entity-level agreement (F1 between annotators), discuss disagreements and update the guideline
   before splitting the rest of the work.
4. **Rule confidence** is not invented: once gold train/validation data exists, each rule's precision is
   measured on it and stored as that rule's confidence.

## 5. Roadmap

| Stage | Deliverable | Status |
|---|---|---|
| 1 | Folder structure, download, filtering, taxonomy, EDA | done |
| 2 | Preprocessing (`process_ingredient_text`), tokenisation with character offsets, validation checks | done |
| 3 | Entity schema + guideline, dictionaries, regex rules, weak labeller, silver dataset, entity-label EDA | done |
| 4 | Gold sampling, Doccano export/import, agreement script | tooling done: **the team now annotates** |
| 5 | Grouped (leakage-free) train/validation/test split, HF-ready files, `label2id.json` | done |
| 6 | Dictionary baseline evaluated on gold test: overall + per-class P/R/F1 | code done: run after gold |
| 7 | Error analysis CSV and limitations section | code done: run after gold |
| P2 | CRF, DistilBERT (+augmentation, +CRF), BERT; comparison on silver / OCR-noisy test; exported model | done (see README Part 2) |
| P3 | Knowledge base, 60 real packet photos, EasyOCR pipeline, entity linking, scanner, Streamlit app, Docker | done (see README Part 3) |
