# Improving our CRNN on real packet crops

Same 377 real test crops, two-fold split by photo. 494 Open Food Facts crops added to training.

| Reader | CER | CER (ignoring case) | WER | Lines exact |
|---|---|---|---|---|
| CRNN v2 + fine-tuned (before) | 0.145 | 0.113 | 0.440 | 17.8% |
| + Open Food Facts crops | 0.115 | 0.091 | 0.379 | 22.5% |
| + Open Food Facts crops + beam search with food LM | 0.092 | 0.067 | 0.277 | 34.2% |
| RapidOCR (reference) | 0.069 | 0.040 | 0.221 | 50.7% |

Beam-search settings (chosen on the other half): alpha 0.7, beta 0.5; alpha 0.7, beta 1.0

## Lines changed by beam search

| Label | Greedy | Beam + LM |
|---|---|---|
| `Multi Grain Flour Mix (Rice Flour, Whole Oats Flour,` | `IgMult Grain Flour Miax (Rice Flour, Wihole Oats Flour,` | `Multi Grain Flour Mix (Rice Flour, Whole Oats Flour,` |
| `REFINED WHEAT FLOUR (MAIDA) (73%), SUGAR, REFINED PALM OIL, INVERT SUGAR SYRUP, MILK PRODUCTS (MILK` | `FI WHEAT FLOUR (mAIDA) (73%), SUGAR, REFINED PALM OIL, INVERT SUGAR SYRUP mILK PRODUCTS (MILK` | `WHEAT FLOUR (mAIDA) (73%), SUGAR, REFINED PALM OIL, INVERT SUGAR SYRUP mILK PRODUCTS (MILK` |
| `SOLIDS&SWEETENED CONDENSED PARTLY SKIMMED MILK), RAISING AGENTS [503(ii) & 500(ii)], IODISED SALT, MINERALS**, ARTIFICIAL` | `SOLIDS & SWEETENED CONDENSED PARTUY SKIMMED MILK), RAISING AGENTS [503(i) & 500(ii)], IODISED SALT, MINERALS**, ARTIFICIAL` | `SOLIDS & SWEETENED CONDENSED PARTLY SKIMMED MILK), RAISING AGENTS [503(ii) & 500(ii)], IODISED SALT, MINERALS**, ARTIFICIAL` |
| `Powder (28%), Chickpea Flour (18%), Maize Starch,` | `Powder(28%), Chickpea Flour(18%), Maize Starch,` | `Powder (28%), Chickpea Flour(18%), Maize Starch,` |
| `Edible Vegetable Oil (Palmolein Oil), Dried Potato` | `Eoiole Vegetable Oil (Palmolein Oil), Dried Potato` | `Edible Vegetable Oil (Palmolein Oil), Dried Potato` |
| `iodised Salt, Dried Mango Powder, Red Chilli` | `lodised Salt, Dried Mango Powder, Red Chillil` | `lodised Salt, Dried Mango Powder, Red Chilli` |
| `Maltodextrin, Mint Flavor(Mint, Garlic, Tamarind),` | `Matodextrin, Mint Flavor (Mint, Garlic, Tamarind),` | `Maltodextrin, Mint Flavor (Mint, Garlic, Tamarind),` |
| `Powder, Onion Powder, Cardamom, Ginger Powder,` | `Powder, Onion Powder, Cardamomn, Ginger Powder,` | `Powder, Onion Powder, Cardamom, Ginger Powder,` |
| `CONCENTRATE#,` | `CONCENTRATE` | `CONCENTRATE,` |
| `(TRISODIUM CITRATE AND CITRIC ACID),` | `(TRISODIUM CITFATE AND CITRIC ACID),` | `(TRISODIUM CITRATE AND CITRIC ACID),` |
| `FLAVOURS (NATURE-IDENTICAL AND NATURAL),` | `FLAVOURS (NATURE-ADENTICAL AND NATURAL],` | `FLAVOURS (NATURE-IDENTICAL AND NATURALT,` |
| `CARRAGEENAN), ANTIOXIDANT (ASCORBIC ACID)` | `CARRAGEENAN; ANTIOXIDANT (ASCOREIC ACID),` | `CARRAGEENAN), ANTIOXIDANT (ASCORBIC ACID),` |
| `AND VITAMIN D2 (ERGOCALCIFEROL).` | `AND VITAMIN D2 (ERGOCALCIFEROL).` | `AND VITAMIN D2 (ERGOCALCIFEROL),` |
| `Refined White Flour (Maida), Double Toned Milk (27%), Unsalted Butter, Sugar, Water, Wheat` | `IGREDIENTS: Refined White Flour (Maida), Double Toned Milk* (27%), Unsalted Butter, Sugar, Water, Wheat` | `Refined White Flour (Maida), Double Toned Milk (27%), Unsalted Butter, Sugar, Water, Wheat` |
| `CARBONATED` | `INGREDENTS CARBONATED` | `CARBONATED` |
| `HERBS (QUILLAJA EXTRACT INS` | `HERBS (OUILLAjA EXTRACT [NS` | `HERBS (OUILLAjA EXTRACT INS` |
| `PERMITTED CLASS II` | `PERMITTED CLASS [I` | `PERMITTED CLASS II` |
| `PRESERVATIVES (INS 211)` | `PRESRVATIVES (NS 211),` | `PRESERVATIVES (INS 211),` |
| `PERMITTED NATURAL FOOD` | `PERMIITED NATURAL FOOD` | `PERMITTED NATURAL FOOD` |
| `COLOUR (INS 150d) AND ADDED` | `gOLOUR (INS 1603) AND ADDED` | `COLOUR (INS 1503) AND ADDED` |