# NER dataset card (for Person 2)

## Files

| Path | What | Use it for |
|---|---|---|
| `silver/train.jsonl.gz` | 15.6k products, automatic (silver) labels | **training** |
| `silver/validation.jsonl.gz` | 3.3k products, silver | quick checks / early stopping while gold is not ready |
| `silver/test.jsonl.gz` | 3.5k products, silver | do **not** train on it; same products as the gold test pool |
| `gold/train.jsonl` | human-verified (after annotation) | fine-tuning on clean labels (optional) |
| `gold/validation.jsonl` | human-verified | model selection / early stopping |
| `gold/test.jsonl` | human-verified | **the only numbers to report** |
| `label2id.json`, `id2label.json` | 21 BIO tags | model config |

The `gold/` files appear after the team has annotated and run `python -m src.annotation.import_annotations`.
Silver and gold use the **same grouped split**: a gold test product, and every near-duplicate of it, is
never in silver train.

## Record format (one JSON object per line)

```json
{
  "id": "8901058000290", "product_id": "8901058000290", "country_group": "india", "split": "train",
  "text": "Sugar, Acidity regulator (INS 330)",
  "tokens": ["Sugar", ",", "Acidity", "regulator", "(", "INS", "330", ")"],
  "ner_tags": [1, 0, 15, 16, 0, 13, 14, 0],
  "ner_tag_names": ["B-SUGAR", "O", "B-FUNCTION_CLASS", "I-FUNCTION_CLASS", "O", "B-INS_CODE", "I-INS_CODE", "O"],
  "token_offsets": [[0, 5], [5, 6], ...],
  "entities": [{"start": 0, "end": 5, "text": "Sugar", "label": "SUGAR", "rule": "dict:sugars", "number": ""}, ...],
  "label_source": "silver_weak_supervision"
}
```

## Labels (`label2id.json`)

`O`=0, then `B-`/`I-` pairs in this order: SUGAR (1, 2), SWEETENER (3, 4), FAT (5, 6), PRESERVATIVE (7, 8),
COLOUR (9, 10), ADDITIVE (11, 12), INS_CODE (13, 14), FUNCTION_CLASS (15, 16), FLAVOURING (17, 18),
INGREDIENT (19, 20). Definitions: `reports/entity_schema.md`.

## Tokenisation assumptions

* `tokens` are **word-level** tokens from `src/preprocessing/tokenize.py`, computed on the **normalised**
  `text` (see `src/preprocessing/normalize.py`). One BIO tag per word.
* With Hugging Face tokenizers use `is_split_into_words=True` and label only the **first sub-word** of each
  word (others `-100`), the standard `word_ids()` alignment.
* Case is kept. `distilbert-base-uncased` lowercases internally; a cased model can use capitalisation.
* Lengths: max 377 words in the data, validation limit 400 (median 50). With BERT's 512 sub-word limit, use `truncation=True`.
  Very long lists are rare.

## Loading

```python
import json
from datasets import Dataset, DatasetDict, Sequence, ClassLabel

id2label = {int(k): v for k, v in json.load(open("data/splits/id2label.json")).items()}
names = [id2label[i] for i in range(len(id2label))]
ds = DatasetDict({s: Dataset.from_json(f"data/splits/silver/{s}.jsonl.gz")
                  for s in ("train", "validation", "test")})
ds = ds.cast_column("ner_tags", Sequence(ClassLabel(names=names)))
```

## Evaluation: use the shared scorer

```python
from src.evaluation.metrics import evaluate_tag_sequences, format_report
result = evaluate_tag_sequences(gold_tag_name_lists, predicted_tag_name_lists)   # strict entity-level
print(format_report(result, "DistilBERT on gold test"))
```

Strict entity-level P/R/F1 (exact span + label), micro and per class, the same scorer as the
dictionary baseline. `src/evaluation/error_analysis.compare_entities` produces the same error categories
as the baseline error analysis.

## Known properties of the silver labels

* INGREDIENT (59% of entities) comes entirely from the fallback rule ("a short leftover segment is an
  ingredient"). It is the noisiest class.
* Rare words (in <3 products) are left unlabelled (`O`) to keep OCR garbage out, so silver recall is
  lower for rare ingredients.
* `rule` in each entity tells you which rule produced it, so you can filter or weight by rule family.
* Augmentation: `src/evaluation/noise_robustness.add_ocr_noise(text, rate, rng)` injects OCR-style
  character confusions without changing the text length, so entity spans remain valid.
