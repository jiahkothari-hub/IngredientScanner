# Improving our CRNN on real packet crops

Same 377 real test crops, two-fold split by photo. 494 Open Food Facts crops added to training.

| Reader | CER | CER (ignoring case) | WER | Lines exact |
|---|---|---|---|---|
| CRNN v2 + fine-tuned (before) | 0.145 | 0.113 | 0.440 | 17.8% |
| + Open Food Facts crops | 0.112 | 0.086 | 0.374 | 23.6% |
| + Open Food Facts crops + beam search with food LM | 0.087 | 0.060 | 0.268 | 36.1% |
| RapidOCR (reference) | 0.069 | 0.040 | 0.221 | 50.7% |

Beam-search settings (chosen on the other half): alpha 0.7, beta 1.0; alpha 0.7, beta 1.0

## Lines changed by beam search

| Label | Greedy | Beam + LM |
|---|---|---|
| `Multi Grain Flour Mix (Rice Flour, Whole Oats Flour,` | `IgMulti Grain Flour Miox (Rice Flour, Wihole Oats Flour,` | `Multi Grain Flour Mix (Rice Flour, Whole Oats Flour,` |
| `REFINED WHEAT FLOUR (MAIDA) (73%), SUGAR, REFINED PALM OIL, INVERT SUGAR SYRUP, MILK PRODUCTS (MILK` | `EFINED WHEAT FLOUR (MAIDA) (73%), SUGAR, REFINED PALM OIL, INVERT SUGAR SYRUP, MILK PRODUCTS (MILK` | `REFINED WHEAT FLOUR (MAIDA) (73%), SUGAR, REFINED PALM OIL, INVERT SUGAR SYRUP, MILK PRODUCTS (MILK` |
| `SOLIDS&SWEETENED CONDENSED PARTLY SKIMMED MILK), RAISING AGENTS [503(ii) & 500(ii)], IODISED SALT, MINERALS**, ARTIFICIAL` | `SOLIDS & SWEETENED CONDENSED PARTUY SKIMMED mILK), RAISING AGENTS [503 (ii) & 500 (ii)], IODISED SALT, MINERALS*, ARTIFICIAL` | `SOLIDS & SWEETENED CONDENSED PARTLY SKIMMED mILK), RAISING AGENTS [503(ii) & 500 (ii)], IODISED SALT, MINERALS, ARTIFICIAL` |
| `Powder (28%), Chickpea Flour (18%), Maize Starch,` | `Powder(28%), Chickpea Flour(18%), Maize Starch,` | `Powder (28%), Chickpea Flour (18%), Maize Starch,` |
| `Edible Vegetable Oil (Palmolein Oil), Dried Potato` | `Eoible Vegetable Oil (Palmolein Oil), Dried Potato` | `Edible Vegetable Oil (Palmolein Oil), Dried Potato` |
| `Maltodextrin, Mint Flavor(Mint, Garlic, Tamarind),` | `Maltodextrin, Mint Flavor (Mtint, Garlic, Tamarind)` | `Maltodextrin, Mint Flavor (Mint, Garlic, Tamarind),` |
| `Powder, Onion Powder, Cardamom, Ginger Powder,` | `Powder, Onion Powder, Cardamomn, Ginger Powder,` | `Powder, Onion Powder, Cardamom, Ginger Powder,` |
| `STABILIZERS (CAROB BEAN GUM AND` | `STABILIZERS (CAROB BEAN GUMAND` | `STABILIZERS (CAROB BEAN GUM AND` |
| `CARRAGEENAN), ANTIOXIDANT (ASCORBIC ACID)` | `CARRAGEENAN); ANTIOXIDANT (ASCORBIC ACID),` | `CARRAGEENAN), ANTIOXIDANT (ASCORBIC ACID),` |
| `CARBONATED` | `NREDT CARBNATED` | `INGRED CARBONATED` |
| `ACIDITY REGULATOR (INS 330),` | `ACIDITY REGULATOR AINS 330)` | `ACIDITY REGULATOR (INS 330)` |
| `HERBS (QUILLAJA EXTRACT INS` | `HERBS (OULLAA EXTRACT INS` | `HERBS (OUILLA EXTRACT INS` |
| `PERMITTED CLASS II` | `PERMITTED CLASS [` | `PERMITTED CLASS I` |
| `PERMITTED NATURAL FOOD` | `PERMITTED NATURAL FDOD` | `PERMITTED NATURAL FOOD` |
| `COLOUR (INS 150d) AND ADDED` | `COLOUR (INS 150 3) AND ADDED` | `COLOUR (INS 1503) AND ADDED` |
| `FLAVOURS (NATURAL, NATURE` | `FLAVOUPS (NATURAL, RATURE` | `FLAVOURS (NATURAL, NATURE` |
| `IDENTICAL AND ARTIFICIAL` | `WERTICAL ANb ARTIFICIAL` | `IENTICAL AND ARTIFICIAL` |
| `FLAVOURING SUBSTANCES CRANBERRY)` | `FLAVOURINg SUBSTANCES *` | `FLAVOURINg SUBSTANCES ` |
| `Maltodextrin, lodised Salt, Flavour (Natural and Nature Identical Flavouring` | `Maltodextin, lodised Salt, Flavour (Natural and Nature Identical Flavouring` | `Maltodextrin, lodised Salt, Flavour (Natural and Nature Identical Flavouring` |
| `Substances), Spices & Condiments, Flavour Enhancers (E627 & E631), Edible` | `Substarces), SSpioes & Condiments, Flavour Enhancers (827 & 631), Eoible` | `Substances), Spices & Condiments, Flavour Enhancers (627 & 631), Edible` |