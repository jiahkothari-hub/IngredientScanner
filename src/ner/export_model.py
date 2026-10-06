"""Export the chosen fine-tuned model as a standard Hugging Face model folder.

Usage:  python -m src.ner.export_model --run distilbert_aug

Writes models/ingredient-ner-distilbert/ with:
  config.json (label2id / id2label inside), tokenizer files, model weights as float16 safetensors split
  into shards of <= 45 MB (GitHub refuses files over 100 MB), training_log.json and a model card.

Load it anywhere with the standard API:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    ner = pipeline("token-classification", model="models/ingredient-ner-distilbert", aggregation_strategy="first")
or with the project wrapper (word-level, same output format as the other systems):
    from src.ner.transformer_ner import TransformerTagger
    TransformerTagger("models/ingredient-ner-distilbert").predict("Sugar, INS 330")
"""
import argparse
import json
import shutil

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.utils.config import project_path

EXPORT_DIR = project_path("models/ingredient-ner-distilbert")


def model_card(run: str, log: dict, comparison: dict) -> str:
    rows = comparison.get("summary_f1", {}).get(run, {})
    metrics = "\n".join(f"| {k} | {v:.3f} |" for k, v in rows.items() if k != "sentences_per_second" and v == v)
    return f"""# ingredient-ner-distilbert

DistilBERT (distilbert-base-uncased) fine-tuned for named-entity recognition on food ingredient lists.
Part of the *Ingredient Scanner* student project. Training run: `{run}`.

## Labels
SUGAR, SWEETENER, FAT, PRESERVATIVE, COLOUR, ADDITIVE, INS_CODE, FUNCTION_CLASS, FLAVOURING, INGREDIENT (BIO scheme,
21 tags; see `config.json` and `reports/entity_schema.md`).

## Training data
Silver (rule-generated) labels for Open Food Facts ingredient lists (English; India, UK/Ireland, North America,
other English-speaking markets), seeded subsample of {log['args'].get('max_train')} sentences,
{log['args'].get('epochs')} epochs, learning rate {log['args'].get('lr')}, OCR-noise augmentation:
{log['args'].get('augment')}. Best validation F1 (silver): {log.get('best_val_f1')}.

## Evaluation (entity-level micro F1, strict)
| Evaluation set | F1 |
|---|---:|
{metrics}

`silver_test` uses rule-made labels; `noisy_test_*` adds OCR-style character noise to the same texts.
Human-verified gold results appear in `reports/model_comparison.md` once the gold annotation is imported.

## Intended use and limitations
Identifies and categorises ingredient mentions. It does **not** assess healthiness or safety. Trained on silver
labels, so it inherits their policy and some of their errors. English ingredient lists only. Weights are
stored in float16 to keep the files small; they are loaded as float32 for inference.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="folder name in models/runs/")
    args = parser.parse_args()
    source = project_path(f"models/runs/{args.run}/best")
    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)
    model = AutoModelForTokenClassification.from_pretrained(source, dtype=torch.float32)
    model.half().save_pretrained(EXPORT_DIR, max_shard_size="45MB", safe_serialization=True)
    AutoTokenizer.from_pretrained(source).save_pretrained(EXPORT_DIR)
    if (source / "crf.pt").exists():
        shutil.copy(source / "crf.pt", EXPORT_DIR / "crf.pt")
    log = json.loads((source / "training_log.json").read_text(encoding="utf-8"))
    shutil.copy(source / "training_log.json", EXPORT_DIR / "training_log.json")
    comparison_path = project_path("reports/model_comparison.json")
    comparison = json.loads(comparison_path.read_text(encoding="utf-8")) if comparison_path.exists() else {}
    (EXPORT_DIR / "README.md").write_text(model_card(args.run, log, comparison), encoding="utf-8")
    size = sum(f.stat().st_size for f in EXPORT_DIR.iterdir()) / 1e6
    print(f"Exported {args.run} -> {EXPORT_DIR.relative_to(project_path(''))} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
