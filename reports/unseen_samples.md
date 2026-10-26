# Unseen samples: typed lists and new photos

Produced by `python -m src.evaluation.unseen_samples` (classifier: hybrid). See the module docstring.

## 1. Typed ingredient lists (15 products, hand-written expectations)

| Metric | Value |
|---|---:|
| category recall | 1.0 |
| category precision | 1.0 |
| additive recall | 1.0 |
| additive precision | 0.935 |

| product              | category items found   |   extra category items | additive numbers found   | missed   |
|:---------------------|:-----------------------|-----------------------:|:-------------------------|:---------|
| Nutella              | 2/2                    |                      0 | 1/1                      | -        |
| Pringles Hot&Spicy   | 5/5                    |                      0 | 9/9                      | -        |
| Maggi Masala         | 3/3                    |                      0 | 7/7                      | -        |
| Parle-G              | 5/5                    |                      0 | 5/5                      | -        |
| Cadbury Dairy Milk   | 2/2                    |                      0 | 2/2                      | -        |
| Kurkure Masala Munch | 3/3                    |                      0 | 5/5                      | -        |
| Oreo                 | 3/3                    |                      0 | 3/3                      | -        |
| Coca-Cola            | 3/3                    |                      0 | 2/2                      | -        |
| Diet Coke            | 4/4                    |                      0 | 5/5                      | -        |
| Haribo Goldbears     | 4/4                    |                      0 | 3/3                      | -        |
| Lays Magic Masala    | 4/4                    |                      0 | 4/4                      | -        |
| Kissan Ketchup       | 2/2                    |                      0 | 3/3                      | -        |
| Amul Butter          | 2/2                    |                      0 | 1/1                      | -        |
| Britannia Good Day   | 5/5                    |                      0 | 4/4                      | -        |
| Red Bull             | 4/4                    |                      0 | 4/4                      | -        |

## 2. New real photos (16 products not among the example photos)

Reference = the app's own result on the product's typed ingredient list, so this measures what the photo step loses.

| Metric | Value |
|---|---:|
| photos | 16 |
| category items recovered | 0.556 |
| additive numbers recovered | 0.595 |
| additive numbers precision | 0.532 |

| product                                                       |   rotation |   OCR confidence | section       | category items found   | additive numbers found   |   seconds |
|:--------------------------------------------------------------|-----------:|-----------------:|:--------------|:-----------------------|:-------------------------|----------:|
| Nutella & Go! (Nutella)                                       |          0 |            0.979 | keyword       | 2/2                    | 1/1                      |       3.2 |
| Nutella Ferrero (Nutella,Ferrero)                             |          0 |            0.91  | keyword       | 2/2                    | 1/1                      |       1.8 |
| CLASSIC (Coca-Cola)                                           |          0 |            0.961 | keyword       | 2/2                    | 2/2                      |       2.8 |
| Coca-Cola (Coca-Cola)                                         |          0 |            0.826 | full_text     | 0/1                    | 0/1                      |       6.4 |
| Mother Energy (Mother,Coca-Cola Amatil)                       |         90 |            0.92  | keyword       | 3/4                    | 4/6                      |       3.2 |
| KitKat Dark Chocolate Coated Wafer (Nestle)                   |          0 |            0.987 | keyword       | 4/4                    | 3/3                      |       2.2 |
| KitKat Rich Chocolate Coated Wafer (Nestlé)                   |          0 |            0.85  | comma_density | 3/4                    | 3/3                      |       2.4 |
| Maggi Special Masala (Maggi)                                  |          0 |            0.969 | keyword       | 1/1                    | 0/0                      |       3.9 |
| Special Masala Noodles (Maggi)                                |          0 |            0.975 | keyword       | 0/1                    | 0/0                      |       3.2 |
| Hot & sweet tomato chili sauce (Maggi)                        |        270 |            0.966 | keyword       | 1/1                    | 0/0                      |       1.7 |
| mini oreo (Oreo,โอรีโอ)                                        |          0 |            0.84  | keyword       | 2/4                    | 4/4                      |       6.3 |
| oreo (p) 27.6g (OREO)                                         |          0 |            0.889 | keyword       | 3/5                    | 6/7                      |       5.2 |
| Soda twist (Haribo)                                           |          0 |            0.943 | comma_density | 1/1                    | 0/0                      |       2.4 |
| haribo blue (nan)                                             |          0 |            0.954 | keyword       | 1/1                    | 0/0                      |       3.6 |
| 299 ROP £2.99 P PRINGLES SALT & VINEGAR 18 kcal ri (Pringles) |          0 |            0.893 | full_text     | 0/4                    | 0/6                      |       5.3 |
| Prawn Cocktail Flavour (Pringles)                             |          0 |            0.954 | keyword       | 0/8                    | 1/8                      |       2.7 |
