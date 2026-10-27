# RapidOCR + our CRNN, combined with a food language model

377 real crops from 57 photos. Margins chosen on the other photo half.

CRNN version: improved. Perfect-chooser ceiling: CER 0.0476; CRNN strictly better on 67 lines.

| Reader | CER | CER (ignoring case) | WER | Lines exact |
|---|---|---|---|---|
| RapidOCR alone | 0.069 | 0.040 | 0.221 | 50.7% |
| Our CRNN alone (fine-tuned) | 0.092 | 0.067 | 0.277 | 34.2% |
| Combined (line-level) | 0.077 | 0.042 | 0.235 | 49.6% |
| Combined (word-level) | 0.072 | 0.041 | 0.220 | 51.5% |

Chosen margins: line: inf, 0.1; word: inf, 0.25
Lines changed from RapidOCR's reading (word-level): 31

## Lines where the combination differs from RapidOCR

| Label | RapidOCR | CRNN | Combined |
|---|---|---|---|
| `PERMITTED CLASS II` | `PERMITTED CLASS IL` | `PERMITTED CLASS II` | `PERMITTED CLASS II` |
| `Maltodextrin, lodised Salt, Flavour (Natural and Nature Identical Flavouring` | `Maltodexdrin, lodised Salt, Flavour (Natural and Nature Identical Flavouring` | `maltodextrin, lodised Salt, flavour (Natural and Nature Identical Flavouring` | `maltodextrin, lodised Salt, Flavour (Natural and Nature Identical Flavouring` |
| `Substances), Spices & Condiments, Flavour Enhancers (E627 & E631), Edible` | `Substances), "Spices & Condimenits, Flavour Enhancers (627 & 631), Edible` | `Substances), Spices & Condiments, Flavour Enhancers (327 & 631), Edible` | `Substances), "Spices & Condiments, Flavour Enhancers (627 & 631), Edible` |
| `*As flavouring agent. Contains Onion and Garlic` | `"As filavouring agent. "Contains Onion and Garlic` | `*As flavouring agent. "Contains Onion and Garlic` | `*As flavouring agent. "Contains Onion and Garlic` |
| `ALLERGEN ADVICE: Contains Nut, Milk, Wheat` | `ALLERGEN ADVICE: Contains Nut, Mik, Wheat` | `Allergen ADvice: Contains nut, Milk, Wheat` | `ALLERGEN ADVICE: Contains Nut, Milk, Wheat` |
| `Whole Wheat,` | `WHOLEWHEAT` | `WHOLE WHEAT` | `WHOLE WHEAT` |
| `Refined Sunflower Oil: 99.99%, TBHQ (E319): Antioxidant,` | `Refined Sunflower Oil:99.99%, TBHQ (E319): Antioxidant,` | `Refined Sunflower Oil: 99.99%, TBHQ (E319) Antioxidant,` | `Refined Sunflower Oil: 99.99%, TBHQ (E319) Antioxidant,` |
| `Vitamin A: Micronutrient, Vitamin D: Micronutrient` | `Vitamin A: Micronutrient, Vitamin D: Micronutrient` | `Vitamin A Micronutrient, Vitamin D Micronutrient` | `Vitamin A Micronutrient, Vitamin D Micronutrient` |
| `Powder, Dried Garlic Powder, Black Salt,` | `Powder, DriedGarlic Powder, Black Salt,` | `Powder, Dried Garlic Powder, Black Salt,` | `Powder, Dried Garlic Powder, Black Salt,` |
| `Orange Fruit Powder (0.9%), Colours (171, 110, 102),` | `Fruit Powder (0.9%"), Colours (171, 110, 102),` | `Fruit Powder (0.9%), Colours (171, 110, 102),` | `Fruit Powder (0.9%), Colours (171, 110, 102),` |
| `Vitamins, Anticaking Agent (341), Stabilizers` | `Vitamins, Anticaking Agent (341(i), tabilizers` | `Vitamins, Anticaking Agent (341(ii)), Stabilizers` | `Vitamins, Anticaking Agent (341(i), Stabilizers` |
| `Wheat flour (Atta 53.8%), Refined Palm oil,` | `Wheat flour (Atta 53.8%), Refined Palm oil,` | `Wheat flour (Atta (53.8%), Refssed Palm oil,` | `Wheat flour (Atta (53.8%), Refined Palm oil,` |
| `Sugar, Invert syrup, Multi grain flour mix 2.8% Jowar,` | `Sugar, Invert syrup, Multi grain flour mix 2.8% Jwar,` | `Sugar, Invert syrup, Multi grain flour mix 2.8% (war,` | `Sugar, Invert syrup, Multi grain flour mix 2.8% (war,` |
| `[INS 500(ii), INS 503(ii)], Emulsifier [INS-322(i), INS 472)` | `[INS 500(i),INS 503(i)], Emulsifier [INS-322(i), INS 4720],` | `[INS 500(i), INS 503(ii)], Emulsifier [INS322(i), INS 472),` | `[INS 500(i), INS 503(ii)], Emulsifier [INS322(i), INS 472),` |
| `ALLERGEN: WHEAT, MILK, SOY, SULPHITE` | `ERGEN:WHEAT, MILK, SOY, SULPHITE` | `ERGEN WHEAT, MILK, SOY, SULPHITE` | `ERGEN WHEAT, MILK, SOY, SULPHITE` |
| `Cranberry(4%),Raisins(4%),Almonds(4%),Pumpkin Seeds(3%),Sugar,Cereal` | `anbery(4%), Raisins (4%), Almonds (4%), Pumpkin Seeds` | `le (6%), Raisins (4%), Almonds (4%), Pumpkin Seeds` | `le (6%), Raisins (4%), Almonds (4%), Pumpkin Seeds` |
| `[SUGAR, CITRIC ACID], 10DISED SALT, RAISING AGENTS` | `[SUGAR, CITRIC ACID], IODISED SALT, RAISING AGENTS` | `SUGAR, CITRIC ACID], IODISED SALT, RAISING AGENTS` | `SUGAR, CITRIC ACID], IODISED SALT, RAISING AGENTS` |
| `(503 (ii), 500 (i)], MILK SOLIDS, FLOUR TREATMENT AGENT` | `(503 (). 500 ()], MILK SOLIDS, FLOUR TREATMENT AGENT` | `(503 (Ii), 500 (II)], MILK SOLIDS, FLOUR TREATMENT AGENT` | `(503 (Ii), 500 (II)], MILK SOLIDS, FLOUR TREATMENT AGENT` |
| `[1101 (ii)] AND EMULSIFIER OF VEGETABLE ORIGIN [472e]` | `(1101 () AND EMULSIFIER OF VEGETABLE ORIGIN [472e]` | `(1101 (i)] and emulsifier of vegetaBle origin [472e]` | `(1101 (i)] AND EMULSIFIER OF VEGETABLE ORIGIN [472e]` |
| `(D-GLUCOSE, LEVULOSE)` | `D-GLUCOSE,LEVULOSE` | `0 - Glucose, Levulose` | `0 - Glucose, Levulose` |
| `CONTAINS: WHEAT, MILK` | `CONTAINS:WHEAT,MILK` | `contains wheat, milk` | `contains wheat, milk` |
| `acid), colours (paprika extract, plain` | `acid). coiours (paprika extract, plain` | `acid), colours (poprika extroct, plain` | `acid). colours (paprika extract, plain` |
| `Bacopa monnieri (brahmi) whole plant-BacoMind 20` | `Bacopa monnieri (brahmi) whole plant- Bacolind 2s` | `Bana mornier (brahmi whole plant - Baonlind 2` | `Bana mornier (brahmi whole plant - Baonlind 2` |
| `Ginkgo biloba (ginkgo) leaf - GinkgoSelect` | `Gnkgo biloba (ginkgo) leaf - GinkgoSelect` | `Ginkgo biloba (ginkgo leaf - Ginkgosoloct` | `Ginkgo biloba (ginkgo) leaf - GinkgoSelect` |
| `sd to gtsenosides Rg1, Re, Rf, Rg2, Rb1,` | `d to pesenos des Rg1, Re, Rt, Rg2, Rb1,` | `sed o green ondes Pg), Re, Ri, R2, Rol,` | `sed to green des Pg), Re, Rt, Rg2, Rol,` |
| `Rb2, Rc, Rd 4.25mg` | `Rb2, Rc. Rd 4 25mg` | `602, RC Rd 4 250g` | `602, Rc. Rd 4 25mg` |
| `Made in Australia.` | `Made in Australa.` | `Made n Australia.` | `Made in Australia.` |
| `(5%): Milk Solids, Live aBc Probiotic Cultures` | `(59): Milk Solids. Live aBc Probiotic Cultures` | `(5%), Milk solids, live a Probiotic cultures` | `(5%), Milk Solids. Live aBc Probiotic Cultures` |
| `lactis Lactobacillus casei 1 Billion Probiotic Count` | `6 Lactobacillus casei 1 Billion Probiotic Count` | `c lactobacillus casei 31 Billion Probiotic coun` | `c Lactobacillus casei 1 Billion Probiotic Count` |
| `raising agent (sodium bicarbonate).` | `raising agent (sodlum bicarbonate)` | `raising agent (sodium bicarbonate,` | `raising agent (sodium bicarbonate)` |