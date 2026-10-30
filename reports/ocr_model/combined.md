# RapidOCR + our CRNN, combined with a food language model

377 real crops from 57 photos. Margins chosen on the other photo half.

| Reader | CER | CER (ignoring case) | WER | Lines exact |
|---|---|---|---|---|
| RapidOCR alone | 0.069 | 0.040 | 0.221 | 50.7% |
| Our CRNN alone (fine-tuned) | 0.145 | 0.113 | 0.440 | 17.8% |
| Combined (line-level) | 0.069 | 0.040 | 0.222 | 50.9% |
| Combined (word-level) | 0.069 | 0.040 | 0.220 | 50.7% |

Chosen margins: line: inf, 0.25; word: 1.0, 0.5
Lines changed from RapidOCR's reading (word-level): 11

## Lines where the combination differs from RapidOCR

| Label | RapidOCR | CRNN | Combined |
|---|---|---|---|
| `Maltodextrin, Mint Flavor(Mint, Garlic, Tamarind),` | `Maltodetrin, Mint Rlavor(Mint Garic, Tamarind)` | `Maltodextrin, Mint Flavor(Mint, Garlic, Tamarind),` | `Maltodextrin, Mint Rlavor(Mint Garic, Tamarind)` |
| `*As flavouring agent. Contains Onion and Garlic` | `"As filavouring agent. "Contains Onion and Garlic` | `*As favouring agert. *Contains Onion and Garlic` | `"As favouring agent. "Contains Onion and Garlic` |
| `Vitamins, Anticaking Agent (341), Stabilizers` | `Vitamins, Anticaking Agent (341(i), tabilizers` | `Vitamins, Anticaking Agent (341(ii)), Stabilizers` | `Vitamins, Anticaking Agent (341(i), Stabilizers` |
| `& Artificial Flavouring Substances (Milk & Vanilla).` | `& Artificial Flavouring Substances (Milk & Vanilla).` | `&. Artificial Flavouring Substances (Milk & Vanilla).` | `&. Artificial Flavouring Substances (Milk & Vanilla).` |
| `[1101 (ii)] AND EMULSIFIER OF VEGETABLE ORIGIN [472e]` | `(1101 () AND EMULSIFIER OF VEGETABLE ORIGIN [472e]` | `(1101 (i)] AND EMULSIFIER OF VEGETABLE ORIcIN [472e]` | `(1101 (i)] AND EMULSIFIER OF VEGETABLE ORIGIN [472e]` |
| `(D-GLUCOSE, LEVULOSE)` | `D-GLUCOSE,LEVULOSE` | `D - GluCoSE, LEVULOSE,` | `D - GluCoSE, LEVULOSE,` |
| `acid), colours (paprika extract, plain` | `acid). coiours (paprika extract, plain` | `ocid), colours (popriko extroct, ploin` | `acid). colours (paprika extract, plain` |
| `Potassium Carbonate), Soya Flour, Maize` | `Potasslum Carhonate), Soya Flour, Maize` | `Potassium Curlenate), Soyn Flour, Maize` | `Potassium Carhonate), Soya Flour, Maize` |
| `Acid), Milk Proteins.` | `Acid), MIIk Proteins.` | `Acid), Milk Proteins.` | `Acid), Milk Proteins.` |
| `(5%): Milk Solids, Live aBc Probiotic Cultures` | `(59): Milk Solids. Live aBc Probiotic Cultures` | `(6%), mik solids, live an prohintic cullures` | `(6%), Milk Solids. Live aBc Probiotic Cultures` |
| `lactis Lactobacillus casei 1 Billion Probiotic Count` | `6 Lactobacillus casei 1 Billion Probiotic Count` | `c lactohacilc racpi 1 billin prohiotic coun` | `c Lactobacillus casei 1 Billion Probiotic Count` |