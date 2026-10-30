# Optimiser and hyper-parameter study (CRNN)

Each run: 2500 steps, tough synthetic lines, seed 42, one-cycle schedule without momentum cycling, gradient clipping 5. Selection on real VALIDATION crops (Open Food Facts, no test photos).

| Optimiser | Learning rate | Batch | Synthetic val CER | Real val CER | Minutes |
|---|---|---|---|---|---|
| SGD | 0.01 | 32 | 1.0000 | 1.0000 | 3.2 |
| SGD | 0.03 | 32 | 1.0000 | 1.0000 | 3.3 |
| SGD | 0.1 | 32 | 0.5718 | 0.6946 | 3.2 |
| SGD + momentum | 0.003 | 32 | 1.0000 | 1.0000 | 3.3 |
| SGD + momentum | 0.01 | 32 | 0.6513 | 0.7313 | 3.3 |
| SGD + momentum | 0.03 | 32 | 0.0955 | 0.3437 | 3.4 |
| SGD + Nesterov | 0.003 | 32 | 1.0000 | 1.0000 | 3.3 |
| SGD + Nesterov | 0.01 | 32 | 0.4563 | 0.6341 | 3.3 |
| SGD + Nesterov | 0.03 | 32 | 0.0801 | 0.3440 | 3.4 |
| RMSprop | 0.001 | 16 | 0.1319 | 0.3983 | 2.2 |
| RMSprop | 0.0001 | 32 | 0.8623 | 0.8464 | 3.4 |
| RMSprop | 0.0003 | 32 | 0.1801 | 0.4071 | 3.3 |
| RMSprop | 0.001 | 32 | 0.0709 | 0.3097 | 3.3 |
| RMSprop | 0.001 | 64 | 0.0484 | 0.2821 | 6.4 |
| Adam | 0.0003 | 32 | 0.1880 | 0.4143 | 3.2 |
| Adam | 0.001 | 32 | 0.0767 | 0.3305 | 3.2 |
| Adam | 0.003 | 32 | 0.0672 | 0.3272 | 3.3 |
| AdamW | 0.0003 | 32 | 0.1910 | 0.4268 | 3.3 |
| AdamW | 0.001 | 32 | 0.0753 | 0.3411 | 3.2 |
| AdamW | 0.003 | 32 | 0.0687 | 0.3201 | 3.3 |

**Winner:** RMSprop, learning rate 0.001, batch 64 (real val CER 0.2821).

## Final test: 377 real crops (two-fold by photo, after the full real-photo pipeline)

| CRNN | Greedy CER | + beam search CER | + beam, ignoring case |
|---|---|---|---|
| Current (AdamW 1e-3, momentum cycled) | 0.1153 | 0.0918 | 0.0674 |
| Tuned (RMSprop 0.001, batch 64) | 0.1122 | 0.0868 | 0.0596 |
| RapidOCR (reference) | 0.0689 | | 0.0401 |