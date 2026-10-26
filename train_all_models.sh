#!/bin/bash
# Person 2: train and compare every NER system, then export the chosen model.
# CPU-only machine: the jobs run ONE AFTER ANOTHER (PyTorch slows down dramatically when cores are shared).
# Total on a 4-core laptop CPU: roughly 2 hours.   Usage:  bash train_all_models.sh
set -e
cd "$(dirname "$0")"
python -m src.ner.pretrained distilbert-base-uncased bert-base-cased            # base weights (~700 MB)
python -m src.ner.crf_baseline                                                   # classic CRF (~2 min)
python -m src.ner.transformer_ner --model distilbert-base-uncased --run distilbert     --epochs 2 --max-train 8000
python -m src.ner.transformer_ner --model distilbert-base-uncased --run distilbert_aug --epochs 2 --max-train 8000 --augment
python -m src.ner.transformer_ner --model distilbert-base-uncased --run distilbert_crf --epochs 2 --max-train 8000 --crf
python -m src.ner.transformer_ner --model bert-base-cased         --run bert           --epochs 2 --max-train 8000 --lr 3e-5
python -m src.ner.evaluate_models                                                # comparison tables + figures 12-13
python -m src.ner.export_model --run distilbert_aug                              # -> models/ingredient-ner-distilbert
python -m src.ocr.evaluate_ocr                                                   # end-to-end on the 60 real photos
