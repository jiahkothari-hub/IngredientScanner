# Our fine-tuned CRNN vs RapidOCR (same real crops)

| | RapidOCR (pretrained) | Our CRNN v2 + fine-tuning |
|---|---|---|
| Character error rate | 0.069 | 0.145 |
| CER ignoring case | 0.040 | 0.113 |
| Word error rate | 0.221 | 0.440 |
| Lines read exactly | 50.7% | 17.8% |
| Perfect crops | 191 | 67 |

Crops only our CRNN read exactly: 9; only RapidOCR read exactly: 133.

Most frequent character substitutions:

- RapidOCR: 'e'->'E' x41, 'a'->'A' x31, 'o'->'O' x30, 't'->'T' x28, 's'->'S' x28, 'r'->'R' x26, 'i'->'I' x25, ','->'.' x24
- Our CRNN: 'a'->'o' x65, 'e'->'E' x34, 's'->'S' x34, 'a'->'A' x28, 't'->'T' x26, '.'->',' x24, 'r'->'R' x22, 'o'->'O' x19

## Crops our CRNN read exactly and RapidOCR did not

| Label | RapidOCR |
|---|---|
| `Whole Wheat Flour), Choco Cream (Sugar, Edible Vegetable Oil` | `Whole Wheat Flour), Choco Cream (Sugar, Edible Vegetable Oi` |
| `Maltodextrin, Mint Flavor(Mint, Garlic, Tamarind),` | `Maltodetrin, Mint Rlavor(Mint Garic, Tamarind)` |
| `CONTAINS ADDED FLAVOUR-NATURAL AND` | `CONTAINS ADDED FLAVOUR-NATURALAND` |
| `PERMITTED CLASS II` | `PERMITTED CLASS IL` |
| `Liquid Glucose, Sugar, Edible Vegetable Oil (Soyabean` | `Liquid Glucose, Sugar,Edible Vegetable Oil (Soyabean` |
| `Oil), lodized Salt, Spices & Condiments (Garlic Powder` | `Oil) lodized Salt, Spices & Condiments Garlic Powder` |
| `SUGAR, REFINED PALM OIL, INVERT SUGAR SYRUP*` | `SUGAR, REFINED PALM OIL, INVERT SUGAR SYRUP` |
| `Acid), Milk Proteins.` | `Acid), MIIk Proteins.` |
| `vinegar solids), Onion powder, Corn maltodextrin,` | `vinegar solids), Onion powder, Corn maltodextrin.` |

## Worst crops for our CRNN

| Label | Ours | RapidOCR |
|---|---|---|
| `PROTEIN,` | `Drotein 4` | `Protein 4` |
| `Wheat Flour` | `TTEETTIT` | `WheatFlout` |
| `JOONSUMER CARE` | `] annmnnmo (nma` | `LOONCUMED CARE` |
| `Turkey breast, water,` | `EN; TURKEY BREPST, NAITE` | `TURKEY BREAST, WHITE` |
| `modified` | `MODIFIED` | `MODIFIED` |
| `Preservative:` | `,TEETNLNEE` | `Preservative` |
| `If product is damaged or looks unusual` | `N AVRAINOA, S, NATNVARRA SR, crAAS VENSSe,` | `If product is damaged or looks unusual` |
| `Wheat Flour,` | `TEDIT,` | `WheatFlout,` |
| `phosphates,` | `PHOSPHATES,` | `PHOSPHATES,` |
| `erythorbate, sodium phosphates,` | `ERYTHORPATE, SODIUM ` | `ERYTHORBATE, SODIUM` |