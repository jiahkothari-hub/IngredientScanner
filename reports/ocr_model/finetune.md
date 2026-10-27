# Closing the synthetic-to-real gap

377 real line crops from 57 packet photos, two-fold split by photo: every fine-tuned reading comes from a model that never saw that photo.

| Recogniser | CER | WER | CER (ignoring case) | Lines exact |
|---|---|---|---|---|
| RapidOCR recogniser (pretrained, PP-OCRv6) | 0.069 | 0.221 | 0.040 | 50.7% |
| CRNN v1 (synthetic) | 0.254 | 0.624 | 0.222 | 10.9% |
| CRNN v2 (tougher synthetic) | 0.194 | 0.526 | 0.163 | 14.9% |
| CRNN v2 + fine-tuned on real crops | 0.145 | 0.440 | 0.113 | 17.8% |

Per photo: fine-tuned CRNN better than RapidOCR on 2 photos, RapidOCR better on 55.

Folds: fold 0 trained on 193 crops, test CER 0.160 on 184; fold 1 trained on 184 crops, test CER 0.129 on 193

## Readings changed by fine-tuning

| Label | Before | After | RapidOCR |
|---|---|---|---|
| `Multi Grain Flour Mix (Rice Flour, Whole Oats Flour,` | `Multi Grain Flour Mix (Rice Flour, Whole Oats Flour,` | `dies: Multi Grain Flour Mix (Rice Flour, Whole Oats Flour,` | `Multi Grain Flour Mix (Rice Flour, Whole Oats Flour,` |
| `Whole Wheat Flour), Choco Cream (Sugar, Edible Vegetable Oil` | `Whole Wheat Flour), Choco Cream (Sugar, Edible Vegetable Ol` | `Whole Wheat Flour), Choco Cream (Sugar, Edible Vegetable Oil` | `Whole Wheat Flour), Choco Cream (Sugar, Edible Vegetable Oi` |
| `(INS 322(i))). Sugar, Cocoa Powder (5.6%), Iodized Salt, Vitamins` | `(INS 322(i))), Sugar, Cocoa Powder (5.6%), Iodized Salt, Vitamins` | `(INS 322(i), Sugar, Cocoa Powder (5.6%), lodized Salt, Vitamins` | `(INS 322(1)), Sugar, Cocoa Powder (5.6%), Iodized Salt, Vitamins` |
| `REFINED WHEAT FLOUR (MAIDA) (73%), SUGAR, REFINED PALM OIL, INVERT SUGAR SYRUP, MILK PRODUCTS (MILK` | `REFINED WHEAT FLOUR (MAIDA) (73%), SUGAR, REFINED PALM OIL, INVERT SUGAR SYRUP MILK PRODUCTS (MILK` | `:REFINED WHEAT FLOUR (MAIDA) (73%), SUGAR, REFINED PALM OIL, INVERT SUGAR SYRUP MILK PRODUCTS (MILK` | `REFINED WHEAT FLOUR (MAIDA) (73%), SUGAR, REFINED PALM OIL, INVERT SUGAR SYRUP, MILK PRODUCTS (MILK` |
| `SOLIDS&SWEETENED CONDENSED PARTLY SKIMMED MILK), RAISING AGENTS [503(ii) & 500(ii)], IODISED SALT, MINERALS**, ARTIFICIAL` | `SOLIDS & 5WEETENED CONDENSED PARTUY SKIMMED MILK), RAISING AGENTS [503(i) & 500(ii)], IODISED SALT, MINERALS**, ARTIFICIAL` | `SOLIDS & SWEETENED CONDENSED PARTUY SKIMMED MILK), RAISING AGENTS [503(i) & 500(ii)], IODISED SALT, MINERALS**, ARTIFICIAL` | `SOLIDS & SWEETENED CONDENSED PARTLY SKIMMED MILK), RAISING AGENTS [503(ii) & 500(ii)], IODISED SALT, MINERALS**, ARTIFICIAL` |
| `(MILK & VANILLA) & NATURE IDENTICAL FLAVOURING SUBSTANCES, EMULSIFIER (472e), DOUGH CONDITIONER (223), VITAMINS",` | `(MILK & VANILLA) & NATURE IDENTICAL FLAVOURING SUBSTANCES, EMULSIFIER (4720), DOUGH (ONDITIONER (223), VITAMINS` | `(MILK & VANILLA) & NATURE IDENTICAL FLAVOURING SUBSTANCES, EMULSIFIER (472), DOUGH CONDITIONER (223), VITAMINS` | `(MILK & VANILLA) & NATURE IDENTICAL FLAVOURING SUBSTANCES, EMULSIFIER (472e), DOUGH CONDITIONER (223), VITAMINS` |
| `COLOUR (150d) AND IMPROVER (1101(i))` | `COLOUR (150d) AND IMPROVER (1101(i))` | `COLOUR (150d) AND IMPROVER (1101(i),` | `COLOUR (150d) AND IMPROVER (1101(i)).` |
| `Edible Vegetable Oil (Palmolein Oil), Dried Potato` | `Eoible Vegetable Oil (Palmolein Oil), Dried Potato` | `Eoible Vegetable Oil (Palmolein Oil), Dnied Potato` | `Edible Vegetable Oil (Palmolein Oil). Dried Potato` |
| `iodised Salt, Dried Mango Powder, Red Chilli` | `lodised Salt, Dried Mango Powder, Red Chill` | `lodised Salt, Dried Mango Powder, Red Chili` | `lodised Salt, Dried Mango Powder, Red Chill` |
| `Maltodextrin, Mint Flavor(Mint, Garlic, Tamarind),` | `Matodextrin, Mint Flavor (Mint, Garlic, Tamarind),` | `Maltodextrin, Mint Flavor(Mint, Garlic, Tamarind),` | `Maltodetrin, Mint Rlavor(Mint Garic, Tamarind)` |
| `NATURE IDENTICAL FLAVOURING` | `NATURE IDENTICAL FLAVOURING` | `NATURE IDENTICALFLAVOURING` | `NATURE IDENTICALFLAVOURING` |
| `Powder, Onion Powder, Cardamom, Ginger Powder,` | `Powder, Onion Powder, Cardamorn, Ginger Powder,` | `Powder, Onion Powder, Cardamom, Ginger Powder,` | `Powder, Onion Powder, Cardamom, Ginger Powder,` |
| `CARRAGEENAN), ANTIOXIDANT (ASCORBIC ACID)` | `CARRAGEENAN); ANTIOXIDANT (ASCORBIC ACID,` | `CARRAGEENAN; ANTIOXIDANT (ASCOREIC ACID,` | `CARRAGEENAN), ANTIOXIDANT (ASCORBIC ACID)` |
| `Refined White Flour (Maida), Double Toned Milk (27%), Unsalted Butter, Sugar, Water, Wheat` | `Refined White Flour (Maida), Double Toned Milk* (27%), Unsalted Butter, Sugar, Water, Wheat` | `IGEDIENTS: Refined White Flour (Maida), Double Toned Milk* (27%), Unsalted Butter, Sugar, Water, Wheat` | `Refined White Flour (Maida), Double Toned Milk* (27%), Unsalted Butter, Sugar, Water, Wheat` |
| `HERBS (QUILLAJA EXTRACT INS` | `HERBS (OUILL4/& EXTRACT [NS` | `HERBS (OUILL/A EXTRACT INS` | `HERBS (QUILLAJA EXTRACT INS` |
| `PERMITTED CLASS II` | `PERMITTED (CLASS [I` | `PERMITTED CLASS II` | `PERMITTED CLASS IL` |
| `PERMITTED NATURAL FOOD` | `PERMIITED NATURAL FO0D` | `PERMIITED NATURAL E0oD` | `PERMITTED NATURAL FOOD` |
| `COLOUR (INS 150d) AND ADDED` | `COLOUR (INS 1502) AND ADDED` | `COLOUR INS 1503) AND ADDED` | `COLOUR (INS 1503) AND ADDED` |
| `FLAVOURS (NATURAL, NATURE` | `FLAVOUPS (NATURAL, NATURE` | `FLAVOUPS (NATURAL, "NATURE` | `FLAVOURS (NATURAL, NATURE` |
| `IDENTICAL AND ARTIFICIAL` | `IDERNTICAL AND ARTIFI(CIAL` | `ERICAL AND ARTIFICIAL` | `IDENTICAL AND ARTIFICIAL` |
| `FLAVOURING SUBSTANCES CRANBERRY)` | `FLAVOURING SUBSTANDES *` | `FLAVOURING SUBSTANCES *,` | `FLAVOURING SUBSTANCES-` |
| `Maltodextrin, lodised Salt, Flavour (Natural and Nature Identical Flavouring` | `Ma*odextrin, lodis2d Sat, Flavur (Natural and Nature lJznt-cal Flavourng` | `Matodextrin, fdisad Sat, Flavour (Natural and Nature antical flavouring` | `Maltodexdrin, lodised Salt, Flavour (Natural and Nature Identical Flavouring` |
| `Substances), Spices & Condiments, Flavour Enhancers (E627 & E631), Edible` | `Substarces), "Spies & Condinerts, Flavur Erhacers (327 & 631), Eoiblg` | `substarces), *Spioes & Condimerts, Flavour Erhancers (327 & 631), Eoible` | `Substances), "Spices & Condimenits, Flavour Enhancers (627 & 631), Edible` |
| `Vegetable Oil (Palm, Sunflower, Peanut), Hydrolysed Vegetable Protein,` | `Veetable Ol (Paln, Sunflower, Peaanut), Hydrolysed Vegetatile Protein,` | `Vegatabile Oil (Paltm, Sunflower, Peanut), Hydrolysed Vegetable Ptotein,` | `Vegetable Oil(Pam, Sunfiower, Peanut), Hydrolysed Vegetable Protein,` |
| `Anticaking Agent (551), Cocoa Solids, Lactose).` | `Ant-cakro 33%rt (551), Cocoa Soids, Lactoce),` | `Antcakno apent (551), Cocoa solds, Lactose),` | `Anticaking agent (551). Cocoa Solids, Lactose).` |