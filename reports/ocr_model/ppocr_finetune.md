# Fine-tuning RapidOCR's recogniser (PP-OCRv6 small) on food-label crops

377 real crops from 57 photos, two-fold split by photo. PyTorch copy agrees exactly with RapidOCR on 306/377 crops before fine-tuning.

| Recogniser | CER | CER (ignoring case) | WER | Lines exact |
|---|---|---|---|---|
| RapidOCR as shipped | 0.069 | 0.040 | 0.221 | 50.7% |
| Same model in PyTorch (our colour crops) | 0.069 | 0.040 | 0.217 | 51.2% |
| RapidOCR recogniser fine-tuned on food labels | 0.072 | 0.043 | 0.217 | 51.2% |
| Our CRNN v2 + fine-tuning (from scratch) | 0.145 | 0.113 | 0.440 | 17.8% |

Per photo: fine-tuned better on 12, RapidOCR better on 18.

Folds (learning rate and steps chosen on validation photos inside the training half; misaligned training crops dropped; test halves unchanged):

- fold 0: 10 of 193 training crops dropped; chosen lr 1e-05, 50 steps (validation CER 0.038 vs pretrained 0.040); test CER 0.093 -> 0.098
- fold 1: 13 of 184 training crops dropped; chosen lr 0, 0 steps (validation CER 0.093 vs pretrained 0.093); test CER 0.043 -> 0.043

## Readings changed by fine-tuning

| Label | RapidOCR | Fine-tuned |
|---|---|---|
| `(INS 322(i))). Sugar, Cocoa Powder (5.6%), Iodized Salt, Vitamins` | `(INS 322(1)), Sugar, Cocoa Powder (5.6%), Iodized Salt, Vitamins` | `(INS 322)(1), Sugar, Cocoa Powder (5.6%), lodized Salt, Vitamins` |
| `SOLIDS&SWEETENED CONDENSED PARTLY SKIMMED MILK), RAISING AGENTS [503(ii) & 500(ii)], IODISED SALT, MINERALS**, ARTIFICIAL` | `SOLIDS & SWEETENED CONDENSED PARTLY SKIMMED MILK), RAISING AGENTS [503(ii) & 500(ii)], IODISED SALT, MINERALS**, ARTIFICIAL` | `SOLIDS & SWEETENED CONDENSED PARTLY SKIMMED MILK), RAISING AGENTS [503(ii) & S00(i)], 1ODISED SALT, MINERALS, ARTIFICIAL` |
| `COLOUR (150d) AND IMPROVER (1101(i))` | `COLOUR (150d) AND IMPROVER (1101(i)).` | `COLOUR (150d) AND IMPROVER (1101(i))` |
| `Edible Vegetable Oil (Palmolein Oil), Dried Potato` | `Edible Vegetable Oil (Palmolein Oil). Dried Potato` | `Edible Vegetable Oil (Palmolein Oil), Dried Potato` |
| `Maltodextrin, Mint Flavor(Mint, Garlic, Tamarind),` | `Maltodetrin, Mint Rlavor(Mint Garic, Tamarind)` | `Maltodextrin, Mint Flavor(Mint, Garlic, Tamarind),` |
| `NATURE IDENTICAL FLAVOURING` | `NATURE IDENTICALFLAVOURING` | `NATURE IDENTICAL FLAVOURING` |
| `CONTAINS ADDED FLAVOUR-NATURAL AND` | `CONTAINS ADDED FLAVOUR-NATURALAND` | `CONTAINS ADDED FLAVOUR-NATURAL AND` |
| `FLAVOURS (NATURE-IDENTICAL AND NATURAL),` | `FLAVOURS (NATURE-IDENTICAL AND NATURAL),` | `FLAVOURS (NATURE-IDENTICAL AND NATURAL)` |
| `Vegetable Oil (Palm, Sunflower, Peanut), Hydrolysed Vegetable Protein,` | `Vegetable Oil(Pam, Sunfiower, Peanut), Hydrolysed Vegetable Protein,` | `Vegetable Oil Pam, Sunfiower, Peanut), Hydrolysed Vegetable Protein,` |
| `Anticaking Agent (551), Cocoa Solids, Lactose).` | `Anticaking agent (551). Cocoa Solids, Lactose).` | `Anticaking agent (551). Cocca Solids, Lactose).` |
| `*As flavouring agent. Contains Onion and Garlic` | `"As filavouring agent. "Contains Onion and Garlic` | `"Asflavourng agent. "Contains Onion and Garlic` |
| `CONTAINS PERMITTED CLASS II` | `CONTAINS PERMITTED CLASS II` | `CONTAINS PERMITTED CLASS I` |
| `Tomato Paste (41%), Water, Onions,` | `Tomato Paste (41%), Water, Onions,` | `Tomato Paste (419), Water, Onions,` |
| `Oil), lodized Salt, Spices & Condiments (Garlic Powder` | `Oil) lodized Salt, Spices & Condiments Garlic Powder` | `Oil), lodized Salt, Spices & Condiments (Garlic Powder` |
| `(0.97%), Mixed Spices, Red Chilli Flakes (0.25%) and` | `(0.979%), Mixed Spices, Red Chilli Rakes (0.25%) and` | `(0.979), Mixed Spices, Red hill Rakes (0.25%) and` |
| `Red Paprika (0.19%)), Acidity Regulators (INS 260,` | `Red Paprika (.19%), Acidity Regulators (INS 260,` | `Red Paprika (0.199)), Acidity Regulators (NS 260,` |
| `Refined Sunflower Oil: 99.99%, TBHQ (E319): Antioxidant,` | `Refined Sunflower Oil:99.99%, TBHQ (E319): Antioxidant,` | `Refined Sunflower Oil: 99.99%, TBHQ (E319): Antioxidant,` |
| `RECIPES SCAN THE OR CODE PLAM` | `RECIPES SCAN THE QR CODE` | `RECIPES SCAN THE OR CODE` |
| `Orange Fruit Powder (0.9%), Colours (171, 110, 102),` | `Fruit Powder (0.9%"), Colours (171, 110, 102),` | `Fruit Powder (0.9%), Colours (171, 110, 102),` |
| `Vitamins, Anticaking Agent (341), Stabilizers` | `Vitamins, Anticaking Agent (341(i), tabilizers` | `Vitamins, Anticaking Agent (341(ii), Stabilizers` |
| `Fruit Powder (0.9%), Colours (171, 110, 102),` | `Fruit Powder (0.9%*). Colours (171, 110, 102).` | `Fruit Powder (0. 9%*). Colours (171, 110, 102).` |
| `Flavour Enhancers (627, 631), Edible Vegetable Oil (Palm, Coconut), Anticaking Agent (551))` | `Flavour Enhancers (627, 631), Edible Vegetable Oil (Palm, Coconut), Anticaking Agent (51)` | `Elavour Enhancers (627, 631), Edible Vegetable Oil (Palm, Coconut), Anticaking Agent (551))` |
| `[INS 500(ii), INS 503(ii)], Emulsifier [INS-322(i), INS 472)` | `[INS 500(i),INS 503(i)], Emulsifier [INS-322(i), INS 4720],` | `[INS 500(i),INS 503(i)], Emulsifier [INS-322(), INS 4720]` |
| `Cranberry(4%),Raisins(4%),Almonds(4%),Pumpkin Seeds(3%),Sugar,Cereal` | `anbery(4%), Raisins (4%), Almonds (4%), Pumpkin Seeds` | `aey (4%), Raisins (4%) Amonds (4%), Pumpki Seeds` |
| `Salt,Vitamins & Minerals,Anti Oxidants(INS 320,INS 307b)` | `Vitanins& Minerals, Antioxidants (INS 320, INS 307b)` | `Vitanins& Minerals, Antioxidants (INS320,INS 307b)` |
| `Potato, Edible Vegetable Oil (Palmolein, Cotton Seed, Corn, Sunflower, Rice Bran),` | `Potato, (dible Vegetable Oil Pamolein, Cotto Sed, Corn, Sunflower, Rice Bran),` | `Potato, dble Vegetable Oil Palmolein, Cottn Seed, Corn, Sunflower, Rice Bran),` |
| `Bengal Gram Flour (Besan), Potato Flakes, Potato Starch, Tapioca Starch, Tepary Beans (Moth Dal)` | `Bengal Gram Flour Besan),Potato Flakes, Potato Starch,Tapioca Starch,Tepary Beans (Moth Dal)` | `Benga Gram Flour (Bean), Potato Flae ,Potato Starch,aioca Starc, Teary Beans (Moth Dal)` |
| `Flour, lodised Salt, Red Chilli Powder, Black Salt, Acidity Regulator (INS 330), Black Pepper, Clove,` | `Flour lodised Salt, Re Cilli Powder, Plack Salt, cidity Regulator INS 330),Black Pepper, Clove,` | `Flour loied Salt e Chilli Powder, Pack Salt, cidity Regulator (NS 330),Black Pepper, Clove,` |
| `Dried Ginger Powder, Dried Garlic Powder, Cumin Powder, Bay Leaf, Nutmeg, Cinnamon, Turmeric,` | `Dried Ginger Powe,Dried Garli Powder, Cumin Powder,BayLeaf, Nutmeg,Cinnamon,Turmerc` | `Dried Ginger Powe,Dried Gari wder, Cumin Powder,Bay Leaf, utmeg,Cinnamon, Turmeric,` |
| `Mint Leaves, Natural Flavouring Substances & Nature Identical Flavouring Substances.` | `Mint Leaves,NauralFlavouring Substances & Nature ldentical Flavouring SUubstances.` | `Mint Leaves Naturl Flavouring Substances &Nature ldentical lvouring Substances.` |