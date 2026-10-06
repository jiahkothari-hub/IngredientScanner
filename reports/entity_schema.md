# Entity schema and annotation guidelines (v1.0)

This document defines the NER task. The weak labeller (`src/labeling/`), the gold annotation and every
model are evaluated against **these rules**. If a rule changes, update this file, the version number,
and re-check the affected gold items.

## 1. The task in one sentence

> Mark every ingredient mention in an ingredient list with **one** of 10 labels.
> The model *identifies* mentions. What an additive *does* is looked up afterwards (interpretation).
> Nobody judges whether an ingredient is healthy.

| Layer | Example | Done by |
|---|---|---|
| Identification | "INS 330" is an `INS_CODE` | NER: dictionary baseline, DistilBERT, BERT |
| Interpretation | INS 330 = citric acid, declared as "acidity regulator" | linking to `additives_reference.csv` |
| Health claim | "this causes …" | **never**: out of scope |

## 2. Labels

| Label | What to annotate | Examples |
|---|---|---|
| `SUGAR` | sugars and syrups added for sweetness | sugar, cane sugar, glucose syrup, dextrose, invert sugar syrup, honey, jaggery, molasses, lactose |
| `SWEETENER` | non-sugar sweeteners, incl. polyols (sugar alcohols) | sucralose, aspartame, acesulfame K, steviol glycosides, sorbitol, maltitol, erythritol |
| `FAT` | oils and fats | palm oil, refined palmolein, sunflower oil, hydrogenated vegetable fat, butter, ghee, cocoa butter, shortening |
| `PRESERVATIVE` | a named substance whose INS number is a preservative (200–259, 280–289, 1105) | sodium benzoate, potassium sorbate, sulphur dioxide, sodium nitrite, calcium propionate |
| `COLOUR` | a named colouring substance (INS 100–199, except 170) | tartrazine, caramel colour, annatto, paprika extract, Red 40, titanium dioxide |
| `ADDITIVE` | any other named substance with an INS/E number | citric acid, soy lecithin, xanthan gum, mono- and diglycerides, sodium bicarbonate, MSG |
| `INS_CODE` | an additive code, in any notation | INS 330, E330, E 330, INS No. 330, E150d, (471), 503(ii) |
| `FUNCTION_CLASS` | a functional class name written on the label | acidity regulator, emulsifier, preservative, colour, raising agent, thickener, stabiliser |
| `FLAVOURING` | flavourings | natural flavour, artificial flavouring substances, nature identical flavouring substances, vanillin |
| `INGREDIENT` | any other food ingredient | wheat flour, milk solids, water, salt, tomato paste, maltodextrin, niacin |

The INS number decides PRESERVATIVE / COLOUR / SWEETENER / ADDITIVE for a named substance. The INS system
groups numbers by main function (Codex CXG 36-1989). Exceptions are listed explicitly in
`configs/lexicons/additive_label_rules.yaml`:
- acids 260–297 are ADDITIVE, not PRESERVATIVE;
- calcium carbonate (170) is ADDITIVE, not COLOUR.

## 3. What is annotated

1. **Every ingredient**, including sub-ingredients inside brackets.
2. **The whole ingredient phrase, without framing.** Descriptive words that belong to the ingredient stay
   inside the span: "**refined palm oil**", "**organic cane sugar**", "**partially hydrogenated soybean oil**".
3. **Alternative names in brackets** get their own entity of the same kind:
   "**Refined wheat flour** (**Maida**)": both INGREDIENT.
4. **Allergen-source notes in brackets** are ingredients too: "**Whey** (**milk**)".

## 4. What is NOT annotated

| Do not annotate | Example (not annotated part in *italics*) |
|---|---|
| Percentages and quantities | sugar *(20.7%)*, *2%* |
| Framing words | *contains less than 2% of:* salt; *ingredients:* sugar |
| Allergen statements and "may contain" | *Contains: milk, soy. May contain traces of nuts.* |
| Storage, contact, nutrition and marketing text | *Keep refrigerated. Freephone 0800 …* |
| Claims | *No artificial colours* |
| Purpose phrases | citric acid *(to protect flavour)* |
| Connectors | sugar *and* salt |
| Text in another language (e.g. the Arabic part of a bilingual label) | |

## 5. Overlapping categories: one label per span

A span gets exactly **one** label. When several seem possible, use the first matching row:

| Priority | Situation | Label |
|---|---|---|
| 1 | it is a code (E330, INS 330, (330)) | `INS_CODE` |
| 2 | non-sugar sweetener | `SWEETENER` |
| 3 | sugar or syrup | `SUGAR` |
| 4 | oil or fat (head word oil/fat/butter/ghee …) | `FAT` |
| 5 | flavouring | `FLAVOURING` |
| 6 | named substance with an INS number | `PRESERVATIVE` / `COLOUR` / `ADDITIVE` (by number) |
| 7 | functional class name | `FUNCTION_CLASS` |
| 8 | anything else that is food | `INGREDIENT` |

Worked examples:

| Text | Annotation |
|---|---|
| Acidity regulator (INS 330) | `FUNCTION_CLASS` "Acidity regulator", `INS_CODE` "INS 330" |
| citric acid | `ADDITIVE` (it has INS 330), even though it is also an acidity regulator |
| Emulsifiers (471, soy lecithin) | `FUNCTION_CLASS`, `INS_CODE` "471", `ADDITIVE` "soy lecithin" |
| Raising agents [503(ii), 500(ii)] | `FUNCTION_CLASS`, `INS_CODE` "503(ii)", `INS_CODE` "500(ii)" |
| potassium sorbate (preservative) | `PRESERVATIVE`, `FUNCTION_CLASS` |
| Colour (150d) | `FUNCTION_CLASS` "Colour", `INS_CODE` "150d" |
| caramel colour | `COLOUR` (a substance, E150) |
| artificial colour / permitted natural colours | `FUNCTION_CLASS` (no substance named) |
| Permitted synthetic food colours (102, 110) | `FUNCTION_CLASS`, two `INS_CODE` |
| maltitol syrup | `SWEETENER` (rule 2 beats the head word "syrup") |
| sorbitol | `SWEETENER` (also has INS 420) |
| mono- and diglycerides of fatty acids | `ADDITIVE` (E471), **not** FAT |
| cocoa butter / butter / ghee | `FAT` |
| peanut butter / almond butter | `INGREDIENT` (a food, not a fat) |
| spearmint oil / lemon oil | `FLAVOURING` (flavouring oils) |
| flavour enhancer | `FUNCTION_CLASS` (an additive class, not a flavouring) |

## 6. INS / E-numbers

* Annotate the code **including** its prefix and sub-type: "INS 330", "E 150d", "INS No. 330", "503(ii)",
  "E160a(ii)". Brackets *around* a code are not part of it: "(**471**)".
* A bare number is a code only when it stands for an additive, typically in brackets after a class name.
  Quantities, years and phone numbers are never codes.
* Several codes are separate entities: "(**407**, **466**, **415**)".
* The code is normalised for linking ("INS 503(ii)" → `503ii`), but the **span** is the text as written.

## 7. Compound ingredients

Annotate the compound and its components separately. NER is flat: spans never overlap.

> **Chocolate** (**sugar**, **cocoa mass**, **cocoa butter**, **emulsifier** (**soy lecithin**))
> → INGREDIENT, SUGAR, INGREDIENT, FAT, FUNCTION_CLASS, ADDITIVE

## 8. Ambiguous ingredients (decided once, for consistency)

| Term | Decision | Reason |
|---|---|---|
| maltodextrin | `INGREDIENT` | a starch-derived carbohydrate, not a sugar by most labelling definitions |
| fruit juice concentrate | `INGREDIENT` | a fruit ingredient, even when it adds sweetness |
| lactose | `SUGAR` | a sugar, usually added as such |
| vitamins and minerals (niacin, riboflavin, iron, folic acid) | `INGREDIENT` | fortification, not additives in our task |
| ascorbic acid / tocopherols | `ADDITIVE` | named with an INS number and mostly used as antioxidants |
| calcium carbonate | `ADDITIVE` | INS 170, but used as a mineral or anti-caking agent, not as a colour |
| gelatin | `INGREDIENT` | a food ingredient on labels |
| saffron, turmeric, paprika (spices) | `INGREDIENT` | spices; their *extracts* (paprika extract, curcumin) are `COLOUR` |
| modified starch | `ADDITIVE` | E14xx |
| milk fat / butter oil | `FAT` | head word fat/oil |
| cream, cheese, milk | `INGREDIENT` | foods that contain fat, not fats |

If you meet a new ambiguous term, write it in the `notes` column of `annotation_tracking.csv`. The team
decides and adds it to this table.

## 9. OCR noise and spelling errors

Annotate what the text *means* if it is clear: "**lodized salt**" → INGREDIENT, "**Milk Solrds**" →
INGREDIENT. Garbage that cannot be read is not annotated.

## 10. Annotation workflow (gold data)

1. Open your file `data/annotations/batches/<your name>.jsonl` in Doccano (see README section 7). It is
   **pre-annotated** with silver labels. These are suggestions and can be wrong.
2. For each product, check **every** span: correct the label, fix the boundaries, delete wrong spans, add
   missed ones. Then mark the document as approved (done).
3. The first 40 items are agreement items that everybody annotates. Annotate them first. Then run
   `python -m src.annotation.agreement` and discuss every disagreement before continuing.
4. Export with *"Export only approved documents"* to `data/annotations/exports/<your name>.jsonl`.

**Pre-annotation bias.** Pre-annotations save time, but they tempt annotators to accept errors. Look at
the text first, then at the suggested spans. Never approve a document you have not fully read.
