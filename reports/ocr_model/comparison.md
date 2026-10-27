# Text-line recognisers: ANN vs CNN vs CRNN

Same synthetic training data, CTC loss, optimiser and schedule for every model. Test lines come from test-split products, rendered in fonts never used for training. CER = character edits / reference characters (greedy CTC decoding).

| Model | Parameters | Steps | Train min | CER clean print | CER phone photo | CER hard photo | WER phone photo | Lines exact (phone) | ms / line |
|---|---|---|---|---|---|---|---|---|---|
| CRNN (CNN + BiLSTM) | 1,266,227 | 12,000 | 15.2 | 0.003 | 0.005 | 0.059 | 0.038 | 92.7% | 1.1 |

## CRNN (CNN + BiLSTM): example readings (phone photo)

| Reference | Prediction |
|---|---|
| `GLYCERIN, PURIFIED` | `GLYCERIN, PURIFIED` |
| `Fat 296kJ/` | `Fat 296kJ/` |
| `For Color.` | `For Color.` |
| `flakes, pumpkin seeds, chia seeds,` | `flakes, pumpkin seeds, chia seeds,` |
| `flavors, proplyene glycol` | `flavors, proplyene glycol` |
| `3,5% currants, candied` | `3,5% currants, candied` |
