#!/bin/bash
# NAT extension (Namysl et al., 2020): compare four DistilBERT training regimes under the SAME budget
# (2,000 training sentences, 2 epochs, seed 42), then evaluate every system on clean, synthetic-noise
# and learned-noise test sets.   Usage:  bash run_nat_experiments.sh      (~2.5-3 h on an 8-core CPU)
set -e
cd "$(dirname "$0")"
export USE_TF=0 USE_TORCH=1
COMMON="--model distilbert-base-uncased --epochs 2 --max-train 2000 --threads 8"

python -m src.evaluation.learned_noise                                                  # noise model from the 60 photos
python -m src.ner.transformer_ner $COMMON --run distilbert_2k                           # no augmentation
python -m src.ner.transformer_ner $COMMON --run distilbert_aug_2k --augment             # hand-made confusion pairs
python -m src.ner.transformer_ner $COMMON --run distilbert_aug_learned_2k --augment --noise learned
python -m src.ner.transformer_ner $COMMON --run distilbert_nat_2k --stability 1.0 --noise learned
python -m src.ner.evaluate_models --limit 1000                                          # all systems x all test sets
