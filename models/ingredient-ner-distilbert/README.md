# ingredient-ner-distilbert

DistilBERT (distilbert-base-uncased) fine-tuned for named-entity recognition on food ingredient lists.
Part of the *Ingredient Scanner* student project. Training run: `distilbert_aug`.

## Labels
SUGAR, SWEETENER, FAT, PRESERVATIVE, COLOUR, ADDITIVE, INS_CODE, FUNCTION_CLASS, FLAVOURING, INGREDIENT (BIO scheme,
21 tags; see `config.json` and `reports/entity_schema.md`).

## Training data
Silver (rule-generated) labels for Open Food Facts ingredient lists (English; India, UK/Ireland, North America,
other English-speaking markets), seeded subsample of 8000 sentences,
2 epochs, learning rate 5e-05, OCR-noise augmentation:
True. Best validation F1 (silver): 0.9444.

## Evaluation (entity-level micro F1, strict)
| Evaluation set | F1 |
|---|---:|
| silver_test | 0.940 |
| noisy_test_0.05 | 0.922 |
| noisy_test_0.10 | 0.900 |

`silver_test` uses rule-made labels; `noisy_test_*` adds OCR-style character noise to the same texts.
Human-verified gold results appear in `reports/model_comparison.md` once the gold annotation is imported.

## Intended use and limitations
Identifies and categorises ingredient mentions. It does **not** assess healthiness or safety. Trained on silver
labels, so it inherits their policy and some of their errors. English ingredient lists only. Weights are
stored in float16 to keep the files small; they are loaded as float32 for inference.
