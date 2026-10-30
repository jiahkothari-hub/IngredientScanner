# Ingredient Scanner: speaker notes

About 18–20 minutes in total, roughly a minute per slide. Times are guides. Lines in *[brackets]* are cues, not things to say.

Suggested split: **Person A** slides 1–6 · **Person B** slides 7–13 · **Person C** slides 14–19.

---

## Slide 1 — Cover: Ingredient Scanner  *(~40 s)*

Good morning everyone. Our project is the Ingredient Scanner.

The idea is simple. You take a photo of the ingredient list on a food packet, and the app tells you what every ingredient is, which ones are sugars, which are additives, what the additive codes mean, and which allergens are there.

*[point at the three chips]* These three examples show what it can do.
- It knows that **dextrose** is a sugar, even though the word "sugar" doesn't appear.
- It decodes **INS 330** into **citric acid**.
- It learned on its own that **sucralose** and **aspartame** do the same job: both are sweeteners.

There's OCR and an app around it, but today we'll focus on the NLP part: the data, the cleaning, the NLP techniques, and the models we compared.

---

## Slide 2 — The core task: label every ingredient  *(~1 min)*

Underneath everything is one NLP task: **named entity recognition**, or NER. NER means finding the important items in a piece of text and putting each one into a category.

*[point at the example]* Here's a typical list: "Sugar, Palm Oil, Acidity Regulator (INS 330), dextr0se, Milk Solids." Our system labels each part:
- Sugar is a **SUGAR**
- Palm Oil is a **FAT**
- "Acidity Regulator" is a **function class**, the word that says *why* something was added
- 330 is an **additive code**
- Milk Solids is an ordinary **ingredient**

Look at "dextr0se". It's written with a zero instead of an "o", which is the kind of mistake OCR makes when reading a photo. It's still labelled SUGAR. Handling messy text like that is a theme you'll see throughout this talk.

We use **10 categories** in total. Once each item is labelled, we look it up in our knowledge base. That's how the code becomes "citric acid", dextrose gets flagged as a hidden sugar, and milk gets flagged as an allergen.

---

## Slide 3 — Dataset: 4.5 million products down to 22,355  *(~1 min 15 s)*

Our data comes from **Open Food Facts**, a free, crowd-sourced database of food products from all over the world. Anyone can add a product and its label.

We started with **4.5 million** products and filtered them step by step. *[walk down the funnel]*
- First, only products sold in English-speaking markets.
- Then only products that actually have an ingredient list. Surprisingly, about 72% don't.
- Then only lists written in English.
- Then a sensible length, between 3 and 1,500 characters.
- Then we removed **103,482 exact duplicates**, because the same product is often added many times.

That left about 405,000 lists, far more than we needed for labelling and training. So we **sampled 22,355 lists, balanced by market**: 7,000 from the UK and Ireland, 7,000 from North America, 4,355 from India, and 4,000 from other English-speaking countries.

We balanced them because labels differ by region. Indian labels use **INS codes**, European ones use **E-numbers**, and American ones use names like "Yellow 5". We wanted the model to see all of these.

The chart uses a **log scale**, because otherwise 22,000 next to 4.5 million would be invisible.

Separately, we have **76 real packet photos** for testing the full app, and a knowledge base of **731 additives**.

---

## Slide 4 — Real lists are short but messy  *(~1 min)*

This is a real ingredient list from our data, an Indian sweet. Look at all the things that make it hard. *[point at each coloured part]*
- The **underscores** around "Milk solids" are how Open Food Facts marks allergens. They're not part of the name, so we have to remove them.
- **Percentages** like "12%": 46.6% of lists have them.
- **Brackets**: a compound ingredient lists its own sub-ingredients inside brackets. 76% of lists use brackets.
- **Additive codes** like E471, E407, E412. These are written only as codes, and they're exactly what we want to find, so cleaning must not destroy them.

Some other numbers: 8.6% of lists contain non-English characters, like accents or other scripts. 7.6% are written entirely in capitals. The typical list is about **230 characters**, roughly **14 ingredients**.

So the text is short, but full of quirks, and each one needs handling before a model sees it.

---

## Slide 5 — Cleaning: unify characters, keep every possible ingredient  *(~1 min 30 s)*

This slide shows our cleaning steps. Each row has a before and after example.

1. **HTML entities.** Text copied from websites contains codes like "&amp;amp;" for "&". We convert them back.
2. **Unicode normalisation, NFKC.** The same-looking character can be stored in different ways. A PDF might store "fl" as one special character, and some text uses wide "full-width" letters. To the computer, those wide letters are not the same as normal "INS". NFKC converts all of these to the standard form.
3. **Quotes, dashes and bullets.** Curly quotes become straight ones, all kinds of dashes become one dash, and bullets become commas.
4. **Invisible characters**, like zero-width spaces, which come from copied or OCR text. We delete them.
5. **Allergen underscores** are removed.
6. **Words split across lines.** OCR often reads "emulsi-" on one line and "fier" on the next. We join them back into "emulsifier".
7. **Extra spaces and repeated commas** are tidied up.
8. **Additive codes.** E-330, INS330 and "(330)" all mean the same thing, so a regular expression turns every spelling into just "330". A bare number in brackets counts only if it's a known additive number and a word like "acidity regulator" appears just before it, so "(12%)" or "(500 g)" is never mistaken for a code.
9. **Tokenisation.** Finally we split the text into tokens: words, numbers and punctuation. Each token remembers its exact character position, so any label can be traced back to the exact place in the original text.

*[point at the "kept on purpose" box]* Just as important is what we **didn't** do. We didn't lower-case, remove stop words, or stem, which are the usual NLP cleaning steps.
- Codes and numbers are the things we're looking for.
- Small words matter: "of" in "of which sugars" is part of a meaningful phrase.
- Stemming would merge "sugars", the nutrition row, with "sugar", the ingredient.
- Our model already handles capital letters itself.

---

## Slide 6 — Making labels: rules labelled 357,934 entities  *(~1 min 30 s)*

To train an NER model you need labelled examples. **No labelled ingredient dataset existed**, and labelling 22,000 lists by hand wasn't possible.

So we used **weak supervision**: we wrote rules and dictionaries that label the data automatically. Labels made this way are called **silver labels**. Labels checked by humans are called gold labels.

Where the rules come from *[point at the cards]*:
- **Dictionaries**: hand-made lists of sugars, sweeteners, fats and flavourings, plus the Open Food Facts list of 731 additives with all their names and synonyms.
- **Head-word rules**: in English, the last word of a phrase usually decides what it is. "Refined palm **oil**" is a fat; "organic cane **sugar**" is a sugar.
- **Additive class from the number**: additive numbers are grouped by function, so 100–199 are colours and 200–299 are preservatives.

To give the labels to a model we use **BIO tagging** *[point at the table]*:
- **B** means the beginning of an entity
- **I** means inside, a continuation of the same entity
- **O** means outside, not an entity

So "glucose syrup" is B-SUGAR, I-SUGAR, and the model knows it's **one** sugar, not two. With 10 categories that gives **21 tags**: B and I for each category, plus O.

In total the rules produced **357,934 labelled entities**.

We split the data into **15,556 training, 3,280 validation and 3,487 test** lists. Products with the same name and brand always go into the same split. Otherwise near-identical products could end up in both training and test, and the scores would look better than they really are.

The honest limitation: silver labels contain mistakes, so we've also prepared a **400-product gold set** for human annotation.

---

## Slide 7 — Ten NLP techniques, each with one job  *(~1 min)*

This slide is our map of the NLP techniques in the project. Each row is a technique, where it's used, and an example. *[go down quickly]*
- **Unicode normalisation and tokenisation** clean and split the text.
- **Weak supervision and BIO tagging** give us training labels without manual work.
- **Named entity recognition** is the core task: labelling every ingredient.
- **Edit distance** fixes spelling mistakes from OCR.
- **Entity linking** connects each name or code to an entry in our knowledge base.
- **Bag of words and TF-IDF** turn lists into numbers so we can find similar products.
- **word2vec** learns what ingredients mean from how they're used.
- **An n-gram language model** measures how predictable, or how messy, the text is.
- **Intent classification and reference resolution** power the question-answering assistant.

The next few slides go through the most important ones.

---

## Slide 8 — The pipeline: photo to explanation in under 7 seconds  *(~1 min)*

This is the full pipeline, from left to right. *[follow the diagram]*

1. The user takes a **photo**.
2. **OCR** (optical character recognition) reads the text from the image. We enlarge small print and fix rotated photos first.
3. **Spelling correction** fixes OCR mistakes, like a zero instead of an "o".
4. **Section extraction** finds the part that starts with "Ingredients" and ignores the brand name, the nutrition table and the address.
5. **Hybrid NER** labels each ingredient using three methods in order:
   - our dictionary rules first, because when they match they're almost always right
   - then the DistilBERT model, for words the rules don't know
   - then fuzzy matching, for anything left
6. **Entity linking** looks each item up in the knowledge base.
7. Finally the app shows categories, allergens and nutrition.

The whole thing takes **1.7 to 6.4 seconds per photo**, measured on 16 new photos, on an ordinary laptop with no GPU.

---

## Slide 9 — Edit distance and entity linking  *(~1 min 15 s)*

Two techniques here.

**Edit distance**, also called Levenshtein distance, is the smallest number of single-letter changes needed to turn one word into another. A change is inserting, deleting or replacing one letter. *[point at the examples]*
- "maltodexrin" → "maltodextrin" is 1 edit: one missing "t".
- "dextr0se" → "dextrose" is 1 edit: zero becomes "o".
- "flavcurenhancers" → "flavour enhancers" is 2 edits: one wrong letter and one missing space.

We use a tool called **SymSpell**, which quickly finds the closest word in our dictionary of 6,407 food words, built from the real ingredient lists. Short words may be corrected by only 1 edit, long words by up to 2. Additive codes and numbers are never changed.

**Entity linking** connects a mention to exactly one entry in our knowledge base. We try four ways, in order:
- **By code**: E330 → citric acid.
- **By exact name**: "dextrose" is in our list.
- **By synonym**: "soy lecithin" is another name for Lecithins, E322. "Yellow 5", the American name, is Tartrazine, E102. That's synonymy: one thing, many names.
- **By fuzzy match**, for small typos: "lecithln" is close enough to "lecithin". We accept it only if the similarity score is at least 88 out of 100, and only within the same category.

The app also shows **how** each item was linked, so the user can tell an exact match from a fuzzy guess.

---

## Slide 10 — Similarity: TF-IDF + word2vec cut nutrition error by 13%  *(~1 min 30 s)*

Here's the problem this solves. Sometimes the photo shows the ingredient list but **no nutrition table**. Then we estimate the nutrition by finding the **most similar product** in our database and using its values. So the question is: how do you measure whether two ingredient lists are similar?

A computer can only compare numbers, so each list is turned into a row of numbers. We compared several ways of doing that:
- **Bag of words**: count each word. Order is ignored.
- **TF-IDF**: like bag of words, but common words like "sugar" or "salt" count less and rare, distinctive words like "hazelnut" or "cocoa" count more. Sharing "sugar" tells you little; sharing "hazelnut" tells you a lot.
- **Jaccard**: shared words divided by all words.
- **Cosine similarity**: the angle between two rows of numbers, so a long list and a short list of the same kind of product still match.
- **Euclidean and Manhattan distance**: plain distances.
- **word2vec**: the average of the ingredient meanings, from the next slide.

*[point at the chart]* The chart shows the average calorie error. **Lower is better.**
- Euclidean and Manhattan are worst, about 108 to 112 kcal off, because plain distance punishes long lists.
- The method the app used before scored 91.5.
- **TF-IDF combined with word2vec is best, at 79.7**, about **13% better**.

We tested on 1,320 products against a database of 6,889. The new method was also better for sugar, fat and salt, and a bootstrap statistical test showed the improvement isn't luck.

If even the best match is too different, below 0.4 similarity, the app gives **no estimate** rather than a wrong one.

---

## Slide 11 — word2vec learned ingredient roles without labels  *(~1 min 30 s)*

word2vec is based on one idea: **you know an ingredient by the company it keeps.** Ingredients that appear in the same places in lists usually do the same job.

How it learns: each ingredient is treated as one token, so "glucose syrup" becomes one word. The model plays a guessing game. We hide one ingredient, show it the three before and three after, and it guesses what's missing. This version is called **CBOW**. After many rounds over **14,066 lists**, every ingredient ends up with a list of **100 numbers**, and ingredients that fit the same gaps get similar numbers.

*[point at the nearest neighbours]* Nobody told the model what anything is, yet:
- sucralose's nearest neighbours are aspartame, saccharin and cyclamate, all artificial sweeteners
- citric acid sits next to lactic, phosphoric and acetic acid
- palm oil sits next to palm kernel oil and coconut oil
- dextrose sits next to sugar

*[point at the chart]* The chart is a **t-SNE map**. t-SNE squeezes the 100 numbers down to 2 so we can draw them. Each dot is an ingredient, and dots close together are used in similar ways. The axes themselves mean nothing. The colours are our dictionary categories, added **after** training to check the result. Sugars, fats, colours and flavourings each form their own clusters, even though the model never saw those categories.

Why it's useful: guessing an ingredient's category from its spelling alone is right **90.5%** of the time. Adding the word2vec information raises that to **93.4%**. A name like "aspartame" doesn't look like a sweetener, but it's *used* like one. We also use word2vec in the nutrition estimate on the last slide, and to place unfamiliar ingredients into sub-groups.

---

## Slide 12 — N-gram language model: perplexity 21.0  *(~1 min 30 s)*

A fair question is why we'd predict the next word when the ingredient list is already printed. **We don't use this model to predict ingredients.** We use it to **measure how normal, or how messy, a piece of text is.**

An **n-gram language model** learns from counts which word usually follows which. A bigram looks at one previous word, a trigram at two. It learns, for example:
- after "glucose" comes "syrup"
- after "palm" comes "oil"
- after "skimmed milk" comes "powder"

One problem: a word pair the model never saw would get probability zero, which breaks the calculation. So we use **smoothing**:
- **Laplace**, or add-1, adds one to every count.
- **Add-k** adds a much smaller number.
- **Interpolation** mixes the trigram, bigram and single-word estimates.

We measure the result with **perplexity**: how surprised the model is by new text. Lower is better. Roughly, it's how many words the model is torn between at each step.

*[point at the chart]*
- Single words only: 193.
- Laplace bigram: 109. Laplace gives away too much probability to word pairs it never saw.
- Add-k bigram: 34.6, three times better.
- **Interpolated trigram: 21.0**, the best. Its mixing weights were chosen on the validation set, never the test set.

The most useful result is this one: the typed ingredient lists of our photo products score **24.7**, but the same lists **read by OCR** from the photos score **66.9**. Same products, same ingredients, but OCR text is almost **three times less predictable**. That puts a number on how much noise OCR adds, and it's why we need spelling correction and noise-robust training. A natural next step would be to use this model to choose between spelling corrections.

---

## Slide 13 — The assistant: 91.7% correct on fresh questions  *(~1 min 30 s)*

The app also has an "Ask about it" tab, where you can ask questions about the scanned product.

It's important to say what kind of system this is. It's **not a generative chatbot** like ChatGPT. It's a **retrieval-based question-answering system**: every answer is built from our knowledge base or the scan result, and it shows its source. It can't make up facts.

When you ask a question, it does four things:
1. **Intent classification**: what kind of question is it? What is it, why is it added, is it vegan, does it contain allergens, and so on. We use TF-IDF features with logistic regression. On our development set the model was right **90.6%** of the time, against **67.2%** for simple keyword rules.
2. **Find the ingredient** the question is about: from a code like E322, a name, or a near-match.
3. **Reference resolution**: *[point at the chat]* in "Is it vegan?", the word "it" refers to the ingredient from the previous question, Lecithins. It got this right in 80% of test cases.
4. **Fill an answer template** with the facts.

**Medical questions are always declined.** "Can I eat this with diabetes?" isn't answered with advice; instead it shows the facts from the label, like which sugars are present. That worked in 100% of cases. Questions unrelated to food are also detected and declined.

To test it honestly, we wrote a **fresh set of 24 questions after all our fixes**, and **91.7%** of answers contained every required fact. On that fresh set, rules and model tied on intent, both at 83.3%, so the model's advantage was smaller there.

---

## Slide 14 — Six NER systems compared  *(~1 min 15 s)*

Now the models. We compared six systems for the NER task, from simple to complex. *[go down the table]*
1. **Dictionary rules**: our baseline. They also made the training labels.
2. **CRF**, a conditional random field: a classical machine-learning model for sequences. It uses hand-made features, like whether a word is capitalised, its first and last letters, and its neighbouring words, and it learns which tag tends to follow which.
3. **BERT-base**: a large pretrained transformer with 110 million parameters, fine-tuned to tag each word.
4. **DistilBERT**: a smaller, faster version of BERT, about 40% smaller, that keeps most of its accuracy.
5. **DistilBERT + CRF**: DistilBERT with a CRF layer on top, which only allows valid tag sequences. For example, an I-SUGAR can't come straight after an O.
6. **DistilBERT + noise**: DistilBERT trained on text where, in each round, half of the lists have OCR-style mistakes added, like "l" swapped with "1" or "O" with "0".

Training settings: 8,000 sentences, 2 epochs, the AdamW optimiser, learning rate 5 × 10⁻⁵.

One detail: BERT models split rare words into pieces, called **WordPiece**. "Maltodextrin" becomes "malt", "ode", "xt", "rin". We put the label on the first piece.

---

## Slide 15 — Results: noise-trained DistilBERT keeps 0.90 F1  *(~1 min 30 s)*

We measure with **F1**, which balances two things:
- **precision**: of the entities we found, how many were right
- **recall**: of the real entities, how many we found

We use the strict version: an entity counts only if both its exact span and its category are correct.

*[point at the table]* On **clean text**, every model scores between 0.93 and 0.95. The rules score a perfect 1.0, but only because the rules made the labels, so that's by construction. Clean text can't really separate the systems.

The real test is **noisy text**, because our input comes from OCR. *[point at the bars]* At **10% noise**, meaning one character in ten is wrong:
- plain DistilBERT drops to **0.719**, even below the rules at 0.734
- the bigger BERT model doesn't fix it, and neither does adding a CRF layer
- **DistilBERT trained with noise keeps 0.900**, about 17 points above the rules

So the main lesson is: **robustness came from the training data, not from a bigger model.**

The cost is speed. The rules process about 1,100 sentences per second, DistilBERT about 45, and BERT about 14. That's still fast enough, because the app reads one label at a time.

---

## Slide 16 — Robustness and error analysis  *(~1 min)*

*[left chart]* This shows the rules as we add more and more noise, from 0 to 10%. Every category drops, but **additives drop fastest**, from 1.0 to 0.62. Long chemical names have more letters that can be damaged, and an exact dictionary match fails if even one letter is wrong.

*[right side]* Here we looked at all **1,747 mistakes** of our best model on the 5% noise test set and sorted them into types:
- **Extra entity, 824, almost half**: the model labelled something the reference didn't. Many of these are actually reasonable. For example, it labels "INS 500 ii" as one complete code, while the rules split off the "ii".
- **Wrong boundary, 322**: the right item, but the span is too long or too short.
- **Missed, 251**, and **wrong category, 213**.

Colours are especially hard when noise breaks a multi-word name, like "yeIlow 5 lakc" instead of "yellow 5 lake".

This also reminds us of a limitation: the reference labels come from the rules, which slightly favours the rules in every comparison.

---

## Slide 17 — Real photos: word errors cut from 0.89 to 0.30  *(~1 min 15 s)*

All the results so far were on text. This is the test on **60 real packet photos**, compared with their correct typed ingredient lists.

We compare two versions of the whole system:
- **Version 1** used the EasyOCR library.
- **Version 2** uses a better OCR engine, RapidOCR, enlarges small print, fixes rotation, corrects spelling, and uses the hybrid NER.

*[point at the table]*
- **Word error rate**, the number of word corrections needed divided by the number of correct words, fell from **0.89 to 0.30**.
- **Character error rate** fell from 0.62 to 0.22.
- **Readable photos** rose from 42 to 55 out of 60.
- **Additives found**, measured as F1, rose from 0.64 to **0.87**.

One honest finding: once the text is this clean, the **rules alone do about as well as the hybrid** on real photos, 0.881 against 0.868 for additives. The model's noise robustness matters less here, because most remaining errors aren't misspelled words. They're text the OCR **never read at all**, like a line that was cut off or blurred.

---

## Slide 18 — Building on a research paper: NAT  *(~1 min 30 s)*

For the research part, our base paper is **NAT, Noise-Aware Training**, by Namysl and colleagues, published at ACL 2020. It's about making sequence-labelling models robust to OCR mistakes, and it has two ideas:
1. **Data augmentation**: train on copies of the text with artificial mistakes added.
2. **Stability training**: add a second loss that pushes the model to give **the same prediction** for the clean text and its noisy copy.

*[point at the formula]* The loss is CE plus alpha times KL.
- **CE**, cross-entropy, is the normal training loss on clean text.
- **KL**, the KL divergence, measures how different the model's predictions are on the clean and noisy versions.
- **Alpha** controls how much weight that second part gets.

**Our extension:** the paper uses general noise. We **learned the noise from our own photos**. We compared OCR output with the correct text for 39 photos and counted which characters OCR actually confuses. After correction, about **2.3% of characters** are wrong.

*[point at the chart]* We trained four DistilBERT versions with exactly the same budget: 2,000 sentences, 2 epochs.
- **Stability training was the most robust on every noisy test set.** At 10% noise it reached **0.813**, against **0.758** with no augmentation.
- On **real photos**, though, all four scored almost the same, between 0.750 and 0.756. Same reason as before: the remaining photo errors are missing text, not damaged words.

Two caveats: these are single training runs, so differences under one point are only a trend, and the CRF shown here was trained on four times more data.

---

## Slide 19 — What we would fix next  *(~1 min)*

To finish, the honest limitations and what we'd do next.

1. **Silver labels.** All our NER scores use labels made by our rules, not by people. The most important next step is annotating the 400-product gold set we've already prepared.
2. **English only.** If a packet has Arabic or Hindi text, we keep it but don't understand it. Next would be multilingual OCR and a multilingual model like XLM-RoBERTa.
3. **OCR misses whole lines.** On real photos, more text is lost completely than misspelled. So the next big gains will come from reading more of the label, with a better ingredient-section finder and better guidance for taking the photo, rather than from a better tagger.
4. **Small photo test sets.** We have 60 plus 16 photos. We'd like a bigger photo benchmark and several training runs per model.

That's our project. Thank you. We're happy to take questions, and we can show the app or our notebook live if you'd like.

---

## Quick answers to likely questions

- **Is this NLP and DL?** Yes. The NLP: NER, tokenisation, edit distance, entity linking, TF-IDF, word2vec, n-gram language models, intent classification. The DL: fine-tuned transformers (BERT and DistilBERT), word2vec, and NAT stability training.
- **Isn't training on rule-made labels circular?** Partly, yes. That's why the comparison focuses on noisy text, where the rules break, and why we prepared a gold set.
- **Why not just use the rules?** They fail when a single letter is wrong. At 10% noise they score 0.73 against 0.90 for the noise-trained model.
- **Why DistilBERT and not BERT?** BERT wasn't more accurate here, and it's about three times slower.
- **Is the assistant a chatbot like ChatGPT?** No. It only retrieves facts and fills templates, so it can't invent information, and it declines medical questions.
- **Why the n-gram model if the list is already given?** We use it to measure how predictable the text is. It showed OCR text is about three times harder to predict than typed text.
- **CBOW or skip-gram?** We tried both, with window sizes 2, 3 and 5. CBOW with window 3 was best.
