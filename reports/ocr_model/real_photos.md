# Real packet photos: our CRNN vs RapidOCR

377 text-line crops from 57 photos. Both recognisers read the same crops (found by RapidOCR's detector). Labels come from the typed ingredient lists (see real_photos.py).

| Recogniser | CER | WER | CER (ignoring case) | WER (ignoring case) | Lines exact |
|---|---|---|---|---|---|
| RapidOCR recogniser (PP-OCRv6, pretrained) | 0.069 | 0.221 | 0.040 | 0.174 | 50.7% |
| Our CRNN (trained on synthetic lines) | 0.254 | 0.624 | 0.222 | 0.599 | 10.9% |

Per photo (character error rate): CRNN better on 0 photos, RapidOCR better on 55.

## Crops where the two disagree

| Label | RapidOCR | CRNN |
|---|---|---|
| `Multi Grain Flour Mix (Rice Flour, Whole Oats Flour,` | `Multi Grain Flour Mix (Rice Flour, Whole Oats Flour,` | `Multi Grain Flour Mx (Rice Flour, Whole Oats Flour,` |
| `Whole Wheat Flour), Choco Cream (Sugar, Edible Vegetable Oil` | `Whole Wheat Flour), Choco Cream (Sugar, Edible Vegetable Oi` | `Wnole Wheat Flour), Choco Cream (Sugar, Ed ble Vegetable Oi` |
| `(Sunflower Oil, Palm Oil), Cocoa Solids (63%), Milk Solids, Emulsifier` | `(Sunflower Oil, Palm Oil), Cocoa Solids (6.3%), Milk Solids, Emulsifier` | `(Sunflower Ol, Palm Oil), Cocoa Solids (6.3%), Milk Solids, Emulsfer` |
| `(INS 322(i))). Sugar, Cocoa Powder (5.6%), Iodized Salt, Vitamins` | `(INS 322(1)), Sugar, Cocoa Powder (5.6%), Iodized Salt, Vitamins` | `(INS 322(i))), Sugar, Cocoa Powder (5.6%), lodized Sall, Vtamins` |
| `REFINED WHEAT FLOUR (MAIDA) (73%), SUGAR, REFINED PALM OIL, INVERT SUGAR SYRUP, MILK PRODUCTS (MILK` | `REFINED WHEAT FLOUR (MAIDA) (73%), SUGAR, REFINED PALM OIL, INVERT SUGAR SYRUP, MILK PRODUCTS (MILK` | `REFINED WHEAT FLOUR (MAIDA) (73%), SUGAR, REFINED PALM OIL, INVERT SUGAR SYRUP MILK PRODUCTS (MIL` |
| `SOLIDS&SWEETENED CONDENSED PARTLY SKIMMED MILK), RAISING AGENTS [503(ii) & 500(ii)], IODISED SALT, MINERALS**, ARTIFICIAL` | `SOLIDS & SWEETENED CONDENSED PARTLY SKIMMED MILK), RAISING AGENTS [503(ii) & 500(ii)], IODISED SALT, MINERALS**, ARTIFICIAL` | `50LID5 & SIEETENED (ONDEISED PARID 5RIMMED MILK), RAISING AGENTS [503(ii) & 500(ii)], I0DISED SALT, MINERALS**, ARTIFICIAL` |
| `(MILK & VANILLA) & NATURE IDENTICAL FLAVOURING SUBSTANCES, EMULSIFIER (472e), DOUGH CONDITIONER (223), VITAMINS",` | `(MILK & VANILLA) & NATURE IDENTICAL FLAVOURING SUBSTANCES, EMULSIFIER (472e), DOUGH CONDITIONER (223), VITAMINS` | `[MILK & VANILLA) & NATURE IDENTICAL FLAVOURING SUBSTANCES, EMULSIFIER (472e), DOUGH (ONDITIONER (223), VITAMINS` |
| `COLOUR (150d) AND IMPROVER (1101(i))` | `COLOUR (150d) AND IMPROVER (1101(i)).` | `(OLOUR (150d) AND IMPROVER (1101(i)),` |
| `Edible Vegetable Oil (Palmolein Oil), Dried Potato` | `Edible Vegetable Oil (Palmolein Oil). Dried Potato` | `Eolble Vegetable Oil (Palmolein 0i), Dried Potato` |
| `iodised Salt, Dried Mango Powder, Red Chilli` | `lodised Salt, Dried Mango Powder, Red Chill` | `lodised Sat, Dried Mango Powder, Red Chill` |
| `Maltodextrin, Mint Flavor(Mint, Garlic, Tamarind),` | `Maltodetrin, Mint Rlavor(Mint Garic, Tamarind)` | `Matodetrin, Mint Flavor(Mint, Garlic, Tamarind),` |
| `NATURE IDENTICAL FLAVOURING` | `NATURE IDENTICALFLAVOURING` | `NATURE IDENTICAL FLAVOURING` |
| `CONTAINS ADDED FLAVOUR-NATURAL AND` | `CONTAINS ADDED FLAVOUR-NATURALAND` | `CONTAINS ADDED FLAVOUR-NATURAL AND` |
| `Acidity Regulator (INS 330),` | `Acidity Regulator (INS 330)` | `Acioty Regulator (NS 330)` |
| `Powder, Onion Powder, Cardamom, Ginger Powder,` | `Powder, Onion Powder, Cardamom, Ginger Powder,` | `Powder, Onion Powder, Cardamorn, Ginger Powder,` |
| `Powder, Spice Blend, Black Pepper, Turmeric` | `Powder, Spice Blend, Black Pepper, Turmeric` | `Powder, Spice Blend, Black Peppet, Iurmeric` |
| `FLAVOURS (NATURE-IDENTICAL AND NATURAL),` | `FLAVOURS (NATURE-IDENTICAL AND NATURAL),` | `FLAVOURS (NATURE-IDENTICAL AND NATURAL,` |
| `CARRAGEENAN), ANTIOXIDANT (ASCORBIC ACID)` | `CARRAGEENAN), ANTIOXIDANT (ASCORBIC ACID)` | `CARRAGEENAN); ANTIOXIDANT (ASCORBIC ACD` |
| `CARBONATED` | `CARBONATED` | `CARBDNATED` |
| `ACIDITY REGULATOR (INS 330),` | `ACIDITY REGULATOR (INS 330)` | `ACIDITY REGU AATDR (INS 330)` |
| `HERBS (QUILLAJA EXTRACT INS` | `HERBS (QUILLAJA EXTRACT INS` | `HERBS (DUILL3JA EXTRACT INS` |
| `PERMITTED CLASS II` | `PERMITTED CLASS IL` | `PERMITTED CLA35 Il` |
| `PRESERVATIVES (INS 211)` | `PRESERVATIVES (INS 211)` | `PBESERVATIVE5 (INS 211)` |
| `PERMITTED NATURAL FOOD` | `PERMITTED NATURAL FOOD` | `PERMITTED NATURAL F00D` |
| `COLOUR (INS 150d) AND ADDED` | `COLOUR (INS 1503) AND ADDED` | `C0L0UR (INS 1503) AND ADDED` |