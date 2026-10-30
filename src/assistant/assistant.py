"""The ingredient assistant: answer a question about the scanned product, from facts only.

    from src.assistant.assistant import IngredientAssistant
    bot = IngredientAssistant(result, guide=guide, body_weight_kg=60)   # result = IngredientScanner().scan_*
    reply = bot.answer("Is E471 vegan?")
    reply["text"]       -> "**Mono- and diglycerides of fatty acids** (INS 471) may or may not be vegan: ..."
    reply["sources"]    -> ["Additive database (Open Food Facts taxonomy)"]
    bot.answer("why is it added?", history=[reply])     # "it" = INS 471 (reference resolution)

Pipeline for every question:
  1. intent        what kind of question (src/assistant/intents.py: TF-IDF + logistic regression,
                   with a keyword safety net for medical questions)
  2. mentions      which ingredient(s) it is about (src/assistant/knowledge.py: codes, names, fuzzy names)
  3. reference     "it" / "this one" / "they" with no ingredient named -> the ingredient the previous
                   answer was about (a simple form of pronoun resolution, NLP Unit 6); "this product" ->
                   the scanned product
  4. answer        a template filled with facts from the knowledge base and the scan result, with the
                   source of every fact. Nothing is generated freely, so nothing can be invented.
Medical and health-judgement questions are declined, with the factual information that is relevant.
"""
import re

from src.assistant.intents import classify, rule_intent
from src.assistant.knowledge import (ANIMAL, NOT_VEGETARIAN, REGIONS, STATUS_TEXT, Mention, category_from_question,
                                     knowledge_base, nutrient_from_question)

SRC_KB = "Additive database (Open Food Facts taxonomy)"
SRC_CLASSES = "Function-class definitions (Codex / Open Food Facts)"
SRC_LEXICON = "Ingredient dictionaries (sugars, fats, sweeteners ...)"
SRC_SCAN = "Your scan"
SRC_ADI = "Acceptable daily intakes (JECFA / EFSA)"
SRC_LIMITS = "Daily limits (WHO, EU 1169/2011)"
SRC_ALLERGENS = "14 allergen groups (EU 1169/2011, FSSAI)"
SRC_REGULATORY = "Additive rules: EU, FSSAI (India), FDA (US)"
SRC_SUBGROUPS = "Ingredient groups (name rules + ingredient2vec)"
REGION_WORDS = {"EU": r"\beu\b|europe|e\.u\.", "India": r"india|fssai", "US": r"\bus\b|\busa\b|u\.s\.|america|\bfda\b"}
# words that make a question about food, labels or this product (questions with none of these, no ingredient
# name and nothing to refer back to are out of scope: "what is the speed of light?")
DOMAIN_RE = re.compile(
    r"ingredient|additive|preserv|colou?r|flavou?r|sweet|sugar|syrup|salt|sodium|\bfats?\b|\boils?\b|calor|kcal|"
    r"energy|protein|carb|fib(?:re|er)|vitamin|mineral|vegan|vegetarian|halal|kosher|\bjain|animal|plant.based|"
    r"al+erg|gluten|lactose|dairy|\bnuts?\b|peanut|\bmilk|\beggs?\b|\bsoy|sesame|wheat|label|\bpack|product|food|"
    r"drink|snack|\beat|portion|serving|daily|limit|\badi\b|intake|\bcode|\be\s?-?\d{3}|\bins\b|\b\d{3,4}[a-z]?\b|"
    r"\bbann?(?:ed)?\b|legal|allowed|permitted|approved|\bfda\b|fssai|\beu\b|europe|healthy|\bsafe|contain|hidden|"
    r"e.?numbers?|emulsif|thicken|stabili|raising|acid|gum|starch|wax|extract|chemical|artificial|natural|organic|"
    r"\bdiet\b|nutri|what'?s\s+in\s+(?:this|it|here)|\bin\s+(?:this|here)\b|made\s+(?:of|from|with)", re.IGNORECASE)
PRONOUN_RE = re.compile(r"\b(it|its|it's|this one|that one|that|they|them|those|these)\b", re.IGNORECASE)
PRODUCT_RE = re.compile(r"\b(this|the)\s+(product|pack|packet|food|drink|snack|item|label|one)\b|"
                        r"\b(in|of)\s+(this|it)\b|\bthis\s*\??$", re.IGNORECASE)
ENTITY_INTENTS = {"what_is", "why_added", "diet", "daily_limit", "regulatory"}
CATEGORY_NAMES = {"SUGAR": "sugars", "SWEETENER": "sweeteners", "FAT": "fats and oils", "PRESERVATIVE": "preservatives",
                  "COLOUR": "colours", "ADDITIVE": "other additives", "FLAVOURING": "flavourings",
                  "INGREDIENT": "other ingredients"}
ADDITIVE_LABELS = ("INS_CODE", "ADDITIVE", "PRESERVATIVE", "COLOUR", "SWEETENER")
ALLERGEN_QUESTION_WORDS = {
    "Cereals containing gluten": r"gluten|wheat|coeliac|celiac", "Crustaceans": r"crustacean|shrimp|prawn|crab|lobster",
    "Eggs": r"\beggs?\b", "Fish": r"\bfish\b", "Peanuts": r"peanut|groundnut", "Soybeans": r"\bsoy|soya",
    "Milk": r"milk|dairy|lactose", "Tree nuts": r"\bnuts?\b|almond|hazelnut|cashew|walnut|pistachio",
    "Celery": r"celery", "Mustard": r"mustard", "Sesame": r"sesame", "Sulphites": r"sulph|sulfit",
    "Lupin": r"lupin", "Molluscs": r"mollusc|mussel|oyster|squid"}


def _join(items: list, limit: int = 12) -> str:
    items = list(dict.fromkeys(items))
    shown = ", ".join(items[:limit])
    return shown + (f" and {len(items) - limit} more" if len(items) > limit else "")


class IngredientAssistant:
    def __init__(self, result: dict = None, guide: dict = None, body_weight_kg: float = 60.0):
        self.kb = knowledge_base()
        self.result = result or {}
        self.guide = guide or {}
        self.weight = body_weight_kg
        self.entities = self.result.get("entities", [])

    # ------------------------------------------------------------------------------------ main entry
    def answer(self, question: str, history: list = ()) -> dict:
        question = question.strip()
        intent, confidence = classify(question)
        mentions = self.kb.find_mentions(question, self.entities)
        allergen_words = any(re.search(p, question, re.IGNORECASE) for p in ALLERGEN_QUESTION_WORDS.values()) or \
            re.search(r"allerg|may contain|traces?|free\b", question, re.IGNORECASE)
        if intent == "allergens" and not allergen_words and (mentions or category_from_question(question)):
            intent = "contains"           # "does it contain sodium benzoate / enzymes?" names no allergen
        if not self._in_scope(question, mentions, history, intent):
            intent = "out_of_scope"
        resolved = ""
        product_level = PRODUCT_RE.search(question) or re.search(r"halal|kosher|jain", question, re.IGNORECASE)
        if not mentions and intent in ENTITY_INTENTS and PRONOUN_RE.search(question) and not product_level:
            topic = next((h.get("topic") for h in reversed(list(history)) if h.get("topic")), None)
            if topic:
                kind, key, name = topic
                mentions = [Mention(kind, key, name, "", "reference")]
                resolved = name
        handler = getattr(self, f"_{intent}")
        reply = handler(question, mentions)
        reply.update({"intent": intent, "confidence": round(confidence, 3),
                      "mentions": [m.name for m in mentions], "resolved_reference": resolved,
                      "question": question})
        reply.setdefault("refused", False)
        reply.setdefault("sources", [])
        reply.setdefault("topic", (mentions[0].kind, mentions[0].key, mentions[0].name) if mentions else None)
        discussed = {n.lower() for h in list(history) + [reply] for n in h.get("mentions", [])}
        reply["suggestions"] = self.suggestions(reply, discussed)
        return reply

    # ------------------------------------------------------------------------------------ helpers
    def _in_scope(self, question: str, mentions: list, history, intent: str) -> bool:
        """Is this a question about food, labels or this product? Off-topic questions ("what is the speed of
        light?") have no ingredient name, no food vocabulary, no word from the scanned label and nothing for
        "it" to refer back to."""
        if mentions or intent == "medical" or rule_intent(question) == "smalltalk" or DOMAIN_RE.search(question):
            return True
        from src.kb.ingredient_subgroups import rule_subgroup
        if rule_subgroup(question):          # the ingredient-group rules know many food words ("spirulina")
            return True
        if any(re.search(p, question, re.IGNORECASE) for p in ALLERGEN_QUESTION_WORDS.values()):
            return True
        if any(h.get("topic") for h in history) and PRONOUN_RE.search(question):
            return True
        label_words = {w[:5] for e in self.entities for w in re.findall(r"[a-z]{4,}", e["text"].lower())}
        return any(w[:5] in label_words for w in re.findall(r"[a-z]{4,}", question.lower()))

    def _in_product(self, mention: Mention) -> list:
        """The product's entities that are this mention (same INS number or same name)."""
        if not self.entities:
            return []
        out = []
        for e in self.entities:
            if mention.kind == "entity" and str(self.entities.index(e)) == mention.key:
                out.append(e)
            elif mention.kind == "additive" and (e.get("ins_number") == mention.key or
                                                 (e.get("ins_number") or "").rstrip("iv()") == mention.key):
                out.append(e)
            elif mention.kind in ("term", "class") and mention.key.lower() in e["text"].lower():
                out.append(e)
        return out

    def _where(self, mention: Mention) -> str:
        """Where the ingredient is on this label, and the role the label itself states for it, if any."""
        if not self.entities:
            return ""
        found = self._in_product(mention)
        if not found:
            return ""
        text = f" In this product it appears as “{_join([e['text'] for e in found], 3)}”"
        stated = [e.get("function") for e in found if e.get("function_source") == "label" and e.get("function")]
        return text + (f", where the label says it is used as **{stated[0].lower()}**." if stated else ".")

    def _explain(self, m: Mention) -> tuple:
        """(text, sources) explaining one mention."""
        if m.kind == "additive":
            a = self.kb.additive(m.key)
            code = f"INS {a['number']}" + (f", {a['e_code']}" if a["e_code"] else "")
            what = f"**{a['name']}** ({code}) is a food additive"
            what += f" used as **{a['functions'].lower()}**." if a["functions"] else "."
            desc = f" {a['description']}" if a["description"] else ""
            flags = self.kb.regulatory_flags(m.key)
            note = (f"\n\n⚖️ **Rules differ between regions:** {'; '.join(flags)}. Ask “is it banned?” for details."
                    if flags else "")
            return what + desc + self._where(m) + note, [SRC_KB] + ([SRC_REGULATORY] if flags else [])
        if m.kind == "class":
            c = self.kb.function_class(m.key)
            return f"**{c['name']}**: {c['description']}" + self._where(m), [SRC_CLASSES]
        if m.kind == "entity":
            e = self.entities[int(m.key)]
            label = e["label"]
            if label == "INGREDIENT" and e.get("subgroup") and e["subgroup"] != "other":
                name = e.get("canonical_name") or e["text"]
                return self._subgroup_text(name, e["subgroup"], e.get("subgroup_method") == "embedding"), \
                    [SRC_SCAN, SRC_SUBGROUPS]
            text = f"**{e.get('canonical_name') or e['text']}**"
            category = self.kb.category_text(label)
            text += f" is listed on this label as {CATEGORY_NAMES.get(label, 'an ingredient').rstrip('s')}." \
                if label in CATEGORY_NAMES else " is on this label."
            if e.get("description"):
                text += f" {e['description']}"
            elif category:
                text += f" {category}"
            return text, [SRC_SCAN, SRC_KB]
        category = self.kb.term_category(m.key)
        name = m.name[0].upper() + m.name[1:]
        if not category:                                   # a generic ingredient: which group is it in?
            from src.kb.ingredient_subgroups import subgroup_of
            group, method = subgroup_of(m.name)
            if group != "other":
                return self._subgroup_text(name, group, method == "embedding") + self._where(m), [SRC_SUBGROUPS]
        if category == "SUGAR_LIKE":
            text = self.kb.category_text(category)
            return f"**{name}** is {text[0].lower() + text[1:]}" + self._where(m), [SRC_LEXICON]
        if category in ("SUGAR", "SWEETENER", "FAT", "FLAVOURING", "PRESERVATIVE", "COLOUR", "ADDITIVE"):
            kind = {"SUGAR": "a sugar", "SWEETENER": "a sweetener", "FAT": "a fat or oil", "FLAVOURING": "a flavouring",
                    "PRESERVATIVE": "a preservative", "COLOUR": "a colour", "ADDITIVE": "an additive"}[category]
            return f"**{name}** is {kind}. {self.kb.category_text(category)}" + self._where(m), [SRC_LEXICON]
        return f"**{name}** is a food ingredient." + self._where(m), [SRC_LEXICON]

    def _subgroup_text(self, name: str, group: str, estimated: bool) -> str:
        """'**Chicory root fiber** is a food ingredient in the group Fibres: ...' (+ caffeine guidance)."""
        from src.app.intake import SPECIAL_SUBSTANCES
        from src.kb.ingredient_subgroups import INFO
        info = INFO[group]
        text = (f"**{name}** is a food ingredient in the group **{info['icon']} {info['title']}**"
                + (" (estimated from similar ingredients)" if estimated else "") + f": {info['description']}")
        for words, title, guidance in SPECIAL_SUBSTANCES:
            if any(w in name.lower() for w in words):
                text += f"\n\n**{title}:** {guidance}"
        return text

    def _unknown(self, question: str) -> dict:
        from src.kb.ingredient_subgroups import rule_subgroup
        asked = re.match(r"\s*(?:what\s+is|what'?s|whats|what\s+are|tell\s+me\s+about|explain|define)\s+(?:an?\s+|the\s+)?"
                         r"([a-z][\w\s'-]{1,40}?)\s*\??$", question, re.IGNORECASE)
        term = asked.group(1).strip() if asked else ""
        if term and rule_subgroup(term):     # a generic ingredient the groups know ("caffeine", "spirulina")
            on_label = [e["text"] for e in self.entities if term.lower() in e["text"].lower()]
            where = f" In this product it appears as “{_join(on_label, 3)}”." if on_label else ""
            return {"text": self._subgroup_text(term[0].upper() + term[1:], rule_subgroup(term), False) + where,
                    "sources": [SRC_SUBGROUPS] + ([SRC_SCAN] if on_label else []), "topic": None}
        guesses = self.kb.retrieve(question)
        if guesses:
            text, sources = self._explain(guesses[0])
            return {"text": "I couldn't find that exact name. The closest match in the database is:\n\n" + text,
                    "sources": sources, "topic": (guesses[0].kind, guesses[0].key, guesses[0].name)}
        return {"text": "I couldn't find that ingredient in the database. Try its full name or its INS/E code "
                        "(for example **E471** or **INS 330**).", "sources": []}

    # ------------------------------------------------------------------------------------ intents
    def _what_is(self, question: str, mentions: list) -> dict:
        if not mentions:
            return self._unknown(question)
        parts, sources = [], []
        for m in mentions[:3]:
            text, src = self._explain(m)
            parts.append(text)
            sources += src
        return {"text": "\n\n".join(parts), "sources": list(dict.fromkeys(sources))}

    def _why_added(self, question: str, mentions: list) -> dict:
        if not mentions:
            if PRODUCT_RE.search(question) or PRONOUN_RE.search(question):
                return {"text": "Which ingredient do you mean? Ask for example **“Why is citric acid added?”** or "
                                "tap an ingredient suggestion below.", "sources": []}
            return self._unknown(question)
        parts, sources = [], []
        for m in mentions[:3]:
            if m.kind == "additive":
                a = self.kb.additive(m.key)
                functions = a["functions"] or ""
                classes = self.kb.class_for_function(functions)
                if functions:
                    text = f"**{a['name']}** (INS {a['number']}) is added as **{functions.lower()}**."
                    text += "".join(f" {c['description']}" for c in classes[:2])
                else:
                    text = f"The database doesn't list a function for **{a['name']}** (INS {a['number']})."
                parts.append(text + self._where(m))
                sources += [SRC_KB, SRC_CLASSES]
            elif m.kind == "class":
                c = self.kb.function_class(m.key)
                parts.append(f"A **{c['name'].lower()}** is used because: {c['description']}")
                sources.append(SRC_CLASSES)
            else:
                text, src = self._explain(m)
                parts.append(text)
                sources += src
        return {"text": "\n\n".join(parts), "sources": list(dict.fromkeys(sources))}

    def _diet(self, question: str, mentions: list) -> dict:
        q = question.lower()
        if re.search(r"halal|kosher|jain", q):
            word = re.search(r"halal|kosher|jain", q).group(0)
            text = (f"I can't tell whether a product is **{word}**: that depends on certification and how it was "
                    f"made, which the ingredient list doesn't show. Look for a {word} certification mark on the pack.")
            if self.entities:
                animal = sorted({e["text"] for e in self.entities if ANIMAL.search(e["text"])})
                if animal:
                    text += f" Animal-derived ingredients on this label: {_join(animal)}."
            return {"text": text, "sources": [SRC_SCAN] if self.entities else []}
        vegan = "vegan" in q or "plant" in q or "animal" in q
        diet = "vegan" if vegan else "vegetarian"
        if mentions:
            return self._diet_ingredient(mentions[0], diet)
        if not self.entities:
            return {"text": "Scan a product first, or name an ingredient, e.g. **“Is E120 vegetarian?”**", "sources": []}
        return self._diet_product(diet)

    def _diet_ingredient(self, m: Mention, diet: str) -> dict:
        if m.kind == "additive":
            a = self.kb.additive(m.key)
            status = a[diet]
            name = f"**{a['name']}** (INS {a['number']})"
            if status == "yes":
                text = f"{name} is normally **{diet}**."
            elif status == "no":
                text = f"{name} is **not {diet}**: it comes from animals."
            elif status == "maybe":
                text = (f"{name} **may or may not be {diet}**: it can be made from plant or animal sources, and "
                        f"labels rarely say which. Only the manufacturer can confirm.")
            else:
                text = f"The database has no {diet} information for {name}."
            return {"text": text + self._where(m), "sources": [SRC_KB]}
        name = m.name
        if (NOT_VEGETARIAN.search(name)) or (diet == "vegan" and ANIMAL.search(name)):
            return {"text": f"**{name}** is **not {diet}** (animal-derived).", "sources": [SRC_KB]}
        if diet == "vegetarian" and ANIMAL.search(name):
            return {"text": f"**{name}** is animal-derived but usually **vegetarian** (not vegan).", "sources": [SRC_KB]}
        return {"text": f"**{name}** is normally plant-based or synthetic, so usually **{diet}**. The database has no "
                        f"specific record for it, so check the pack if it matters.", "sources": [SRC_KB]}

    def _diet_product(self, diet: str) -> dict:
        no, maybe = [], []
        for e in self.entities:
            if e["label"] == "FUNCTION_CLASS":
                continue
            number = e.get("ins_number")
            if number and number in self.kb.additives.index:
                status = self.kb.additive(number)[diet]
                name = f"{e.get('canonical_name') or e['text']} (INS {number})"
                if status == "no":
                    no.append(name)
                elif status == "maybe":
                    maybe.append(name)
            elif NOT_VEGETARIAN.search(e["text"]) or (diet == "vegan" and ANIMAL.search(e["text"])):
                no.append(e["text"])
        if no:
            text = f"This product is **not {diet}**: it contains {_join(no)}."
        elif maybe:
            text = (f"No clearly non-{diet} ingredient was found, but {_join(maybe)} **can be plant- or "
                    f"animal-derived**. Check for a {diet} mark on the pack.")
        else:
            text = f"No non-{diet} ingredient was found in this list, so it looks **{diet}**."
        if maybe and no:
            text += f" Also possibly animal-derived: {_join(maybe)}."
        text += " This is based on the ingredients as read; a certification mark on the pack is the reliable check."
        return {"text": text, "sources": [SRC_SCAN, SRC_KB], "topic": None}

    def _contains(self, question: str, mentions: list) -> dict:
        if not self.entities:
            return {"text": "Scan a product first, then I can check what it contains.", "sources": []}
        category = category_from_question(question)
        if any(re.search(p, question, re.IGNORECASE) for p in ALLERGEN_QUESTION_WORDS.values()) and \
                not any(self._in_product(m) for m in mentions):
            return self._allergens(question, mentions)          # "does it contain sulphites / nuts?"
        if not mentions and category:
            return self._list_category(question, mentions)
        if not mentions:
            words = [w for w in re.findall(r"[a-z]{3,}", question.lower())
                     if w not in {"does", "this", "contain", "have", "there", "any", "product", "use", "used"}]
            hits = [e["text"] for e in self.entities if any(w in e["text"].lower() for w in words)]
            if hits:
                return {"text": f"Yes: the label lists {_join(hits)}.", "sources": [SRC_SCAN]}
            return {"text": f"I couldn't find **{' '.join(words) or 'that'}** in this ingredient list. The photo may "
                            f"have missed a word, so check the pack.", "sources": [SRC_SCAN]}
        parts = []
        for m in mentions[:3]:
            found = self._in_product(m)
            if found:
                parts.append(f"**Yes**, it contains **{m.name}**: listed as “{_join([e['text'] for e in found], 3)}”.")
            else:
                parts.append(f"**No**, I couldn't find **{m.name}** in this ingredient list. "
                             f"(If the photo was hard to read, check the pack.)")
        return {"text": "\n\n".join(parts), "sources": [SRC_SCAN]}

    def _allergens(self, question: str, mentions: list) -> dict:
        if not self.result:
            return {"text": "Scan a product first, then I can list its allergens.", "sources": []}
        allergens = self.result.get("allergens") or {}
        contains, may = allergens.get("contains", {}), allergens.get("may_contain", {})
        asked = [g for g, p in ALLERGEN_QUESTION_WORDS.items() if re.search(p, question, re.IGNORECASE)]
        if asked:
            parts = []
            for g in asked:
                if g in contains:
                    parts.append(f"**Yes, it contains {g.lower()}**: “{_join(contains[g], 4)}”.")
                elif g in may:
                    parts.append(f"**{g} is not an ingredient, but the label warns it may contain traces** "
                                 f"(“{_join(may[g], 3)}”).")
                else:
                    parts.append(f"**No {g.lower()}** was found in the ingredient list or the allergy statement.")
            text = " ".join(parts)
        elif contains or may:
            text = ""
            if contains:
                text += "**Contains:** " + "; ".join(f"{g} ({_join(w, 3)})" for g, w in contains.items()) + "."
            if may:
                text += (" " if text else "") + "**May contain (traces):** " + ", ".join(may) + "."
        else:
            text = "None of the 14 major allergen groups was found in this list."
        text += " If you have an allergy, always read the pack itself: OCR can miss words."
        return {"text": text, "sources": [SRC_SCAN, SRC_ALLERGENS], "topic": None}

    def _list_category(self, question: str, mentions: list) -> dict:
        if not self.result:
            return {"text": "Scan a product first, then I can list its ingredients by type.", "sources": []}
        summary = self.result.get("summary", {})
        groups, hidden = summary.get("groups", {}), summary.get("hidden", {})
        roles = [m for m in mentions if m.kind == "class"]
        same_as_category = {"sweetener", "colour", "color", "preservative", "flavouring", "flavoring"}
        if roles and self.kb.function_class(roles[0].key)["name"].lower() in same_as_category:
            roles = []          # "sweeteners": the whole category, not only items whose label states that role
        if roles:                                  # "what flavour enhancers / thickeners are there?"
            role = self.kb.function_class(roles[0].key)["name"]
            items = [f"{e.get('canonical_name') or e['text']}" + (f" ({e['text']})" if e.get("canonical_name") and
                                                                   e["text"] != e["canonical_name"] else "")
                     for e in self.entities if role.lower() in (e.get("function") or "").lower()]
            if items:
                return {"text": f"**{role}s on this label:** {_join(items)}.", "sources": [SRC_SCAN, SRC_KB],
                        "topic": None}
            return {"text": f"No {role.lower()} was found on this label.", "sources": [SRC_SCAN], "topic": None}
        category = category_from_question(question) or "ALL"
        if category.startswith("SUB:"):                    # protein sources, fibres, probiotics ...
            from src.kb.ingredient_subgroups import INFO
            key = category[4:]
            info = INFO[key]
            names = summary.get("subgroups", {}).get(key, [])
            if names:
                listed = _join([n["name"] + (" (estimated)" if n["estimated"] else "") for n in names])
                return {"text": f"{info['icon']} **{info['title']} ({len(names)}):** {listed}. {info['description']}",
                        "sources": [SRC_SCAN, SRC_SUBGROUPS], "topic": None}
            return {"text": f"No {info['title'].lower()} were found among this label's ingredients.",
                    "sources": [SRC_SCAN], "topic": None}
        if category == "hidden_sugar":
            items = hidden.get("sugars_under_other_names", []) + hidden.get("sugar_like_carbohydrates", [])
            text = (f"Sugars under other names: **{_join(items)}**." if items else
                    "No sugars under other names were found: any sugar here is called “sugar”.")
        elif category == "CODE":
            codes = [f"{e['text']} = {e.get('canonical_name') or 'unknown'}" for e in self.entities if e["label"] == "INS_CODE"]
            text = ("Codes on this label: " + "; ".join(codes) + ".") if codes else "No INS/E codes are written on this label."
        elif category == "ADDITIVE":
            items = [e.get("canonical_name") or e["text"] for e in self.entities if e["label"] in ADDITIVE_LABELS]
            text = (f"**{len(set(items))} additives:** {_join(items)}." if items else "No additives were found.")
        elif category == "ALL":
            text = " ".join(f"**{CATEGORY_NAMES[c].capitalize()}:** {_join(v)}." for c, v in groups.items() if v)
        else:
            items = groups.get(category, [])
            name = CATEGORY_NAMES[category]
            text = (f"**{name.capitalize()} ({len(items)}):** {_join(items)}." if items else f"No {name} were found.")
        return {"text": text, "sources": [SRC_SCAN], "topic": None}

    def _portion(self, question: str, mentions: list) -> dict:
        rows = self.guide.get("rows") or []
        if not rows:
            return {"text": "I need nutrition values for that. Open **Is this amount OK?** and type them from the "
                            "pack (or scan the nutrition table), then ask again.", "sources": []}
        unit = "ml" if self.guide.get("drink") else "g"
        portion = self.guide.get("portion", 0)
        key = nutrient_from_question(question)
        row = next((r for r in rows if r["key"] == key), None)
        if row is None:
            from src.app.intake import headline
            text, _ = headline(rows, portion, unit)
            return {"text": text, "sources": [SRC_LIMITS], "topic": None}
        kind = "limit" if row["limit_kind"] == "max" else "reference intake"
        text = (f"A **{portion:g} {unit}** portion has **{row['in_portion']:g} {row['unit']} {row['nutrient'].lower()}**, "
                f"which is **{row['share']:.0%}** of the daily {kind} ({row['daily_value']:g} {row['unit']}).")
        if row["key"] == "sugars_g" and row["in_portion"] > 0:
            text += f" That is about {row['in_portion'] / 4:.1f} teaspoons."
        if row.get("traffic_light"):
            text += f" Per 100 {unit} it is **{row['traffic_light']}** on the UK traffic-light scale."
        return {"text": text, "sources": [SRC_LIMITS], "topic": None}

    def _daily_limit(self, question: str, mentions: list) -> dict:
        from src.app.intake import additive_guidance
        targets = [m for m in mentions if m.kind == "additive"]
        if not targets and self.entities and not mentions:
            entities = [e for e in self.entities if e.get("ins_number")]
        else:
            entities = [{"ins_number": m.key, "canonical_name": m.name, "text": m.name} for m in targets]
        if not entities:
            if mentions:
                return {"text": f"Daily limits (ADIs) exist for additives only, and **{mentions[0].name}** isn't one. "
                                f"For sugars, salt and fat, ask about your portion instead.", "sources": []}
            return {"text": "Name the additive, e.g. **“What is the ADI of aspartame?”**", "sources": []}
        rows = additive_guidance(entities, self.weight)
        if not rows:
            names = _join([e["canonical_name"] for e in entities], 4)
            return {"text": f"There is no specific ADI on record for {names} in our table.", "sources": [SRC_ADI]}
        text = " ".join(f"**{r['additive']}:** {r['daily_limit']} ({r['authority']})." +
                        (f" {r['note']}" if r.get("note") else "") for r in rows[:4])
        return {"text": text + " Labels rarely say how much additive a product contains; permitted use levels "
                               "keep normal portions well below these limits.", "sources": [SRC_ADI]}

    def _regulatory(self, question: str, mentions: list) -> dict:
        """Is an additive banned / restricted in the EU, India or the US? (data/knowledge_base/additive_regulatory.csv)"""
        regions = [r for r, p in REGION_WORDS.items() if re.search(p, question, re.IGNORECASE)] or list(REGIONS)
        targets = [m for m in mentions if m.kind == "additive"]
        if not targets and mentions:
            return {"text": f"Bans and approvals in our table cover food additives only, and **{mentions[0].name}** "
                            f"isn't one.", "sources": []}
        if not targets:
            if not self.entities:
                return {"text": "Name an additive, e.g. **“Is titanium dioxide banned?”**", "sources": []}
            flagged, seen = [], set()
            for e in self.entities:
                number = e.get("ins_number")
                if not number or number in seen:      # "Titanium dioxide (171)" is two entities, one additive
                    continue
                seen.add(number)
                flags = [f for f in self.kb.regulatory_flags(number) if any(f.endswith(r) for r in regions)]
                if flags:
                    name = self.kb.additive(number)["name"] if number in self.kb.additives.index else e["text"]
                    flagged.append(f"**{name}** (INS {number}): {'; '.join(flags)}")
            where = " / ".join(regions)
            if not flagged:
                return {"text": f"None of the additives on this label has a ban, phase-out or warning-label rule "
                                f"recorded for {where} in our table. The table covers the additives whose rules are "
                                f"known to differ between the EU, India and the US.",
                        "sources": [SRC_REGULATORY], "topic": None}
            return {"text": "Additives on this label with different rules elsewhere:\n\n" + "\n\n".join(flagged) +
                            "\n\nAsk about one of them for the details.", "sources": [SRC_REGULATORY], "topic": None}
        parts = []
        for m in targets[:2]:
            status = self.kb.regulatory_status(m.key)
            name = f"**{m.name}** (INS {m.key})"
            if not status:
                parts.append(f"{name} has no ban, phase-out or warning-label rule recorded in our table for the EU, "
                             f"India or the US. The table covers additives whose rules are known to differ between "
                             f"them, so this doesn't prove it is allowed in every food.")
                continue
            lines = [f"{name}:"]
            for r in regions:
                code, note = status[r]
                lines.append(f"- **{r}: {STATUS_TEXT[code]}**" + (f". {note}" if note else ""))
            if status["other"] and len(regions) == len(REGIONS):     # notes on other places, for broad questions
                lines.append(status["other"])
            lines.append(f"(Checked {status['checked']}; rules change, so confirm with the regulator for anything "
                         f"important.)")
            parts.append("\n".join(lines))
        return {"text": "\n\n".join(parts), "sources": [SRC_REGULATORY]}

    def _medical(self, question: str, mentions: list) -> dict:
        text = ("I can't give medical or personal health advice: whether a food suits **you** depends on your health, "
                "and that's a question for a doctor or dietitian.")
        facts = []
        if self.result:
            summary = self.result.get("summary", {})
            groups = summary.get("groups", {})
            if groups.get("SUGAR"):
                facts.append(f"sugars: {_join(groups['SUGAR'], 6)}")
            if groups.get("SWEETENER"):
                facts.append(f"sweeteners: {_join(groups['SWEETENER'], 4)}")
            contains = (self.result.get("allergens") or {}).get("contains", {})
            if contains:
                facts.append(f"allergens: {_join(list(contains), 6)}")
            limited = [r for r in (self.guide.get("rows") or []) if r["limit_kind"] == "max"]
            if limited:
                worst = max(limited, key=lambda r: r["share"])
                facts.append(f"one portion uses {worst['share']:.0%} of the daily {worst['nutrient'].lower()} limit")
        if facts:
            text += " What I can tell you from the label: " + "; ".join(facts) + "."
        return {"text": text, "sources": [SRC_SCAN] if facts else [], "refused": True, "topic": None}

    def _out_of_scope(self, question: str, mentions: list) -> dict:
        asked = re.match(r"\s*(?:what\s+is|what'?s|whats|what\s+are|tell\s+me\s+about|explain|define)\s+(?:an?\s+|the\s+)?"
                         r"([a-z][\w\s'-]{1,40}?)\s*\??$", question, re.IGNORECASE)
        if asked and len(asked.group(1).split()) <= 3:      # could be an ingredient we don't know
            return {"text": (f"I couldn't find “{asked.group(1)}” in our food database. If it's an ingredient, try "
                             f"its full name or its INS/E code. If it isn't about food, I can only answer questions "
                             f"about food ingredients and this product's label."), "sources": [], "topic": None}
        return {"text": ("That's outside what I can help with: I only answer questions about food ingredients and "
                         "this product's label, from our food database. Try asking what an ingredient or code is, "
                         "whether it's vegan, which allergens or hidden sugars are here, or how a portion compares "
                         "with daily limits."), "sources": [], "topic": None}

    def _smalltalk(self, question: str, mentions: list) -> dict:
        if re.search(r"thank|bye|great|ok", question, re.IGNORECASE):
            return {"text": "You're welcome! Ask me anything else about this label.", "sources": [], "topic": None}
        return {"text": ("I answer questions about this product's ingredients from our food database: what an "
                         "ingredient or code is, why it's added, whether it's vegan or vegetarian, allergens, hidden "
                         "sugars, and how a portion compares with daily limits. I don't give medical advice."),
                "sources": [], "topic": None}

    # ------------------------------------------------------------------------------------ suggestions
    def suggestions(self, reply: dict = None, discussed: set = frozenset()) -> list:
        """Up to 3 follow-up questions that make sense for this product (skipping what was discussed)."""
        out = []
        topic = (reply or {}).get("topic")
        intent = (reply or {}).get("intent")
        flagged_topic = topic and topic[0] == "additive" and self.kb.regulatory_flags(topic[1])
        if topic and intent == "what_is":
            out += ["Why is it added?", "Is it banned anywhere?" if flagged_topic else "Is it vegan?"]
        if intent != "regulatory" and not flagged_topic and any(
                self.kb.regulatory_flags(e["ins_number"]) for e in self.entities if e.get("ins_number")):
            out.append("Is anything in this banned elsewhere?")
        names = [e.get("canonical_name") or e["text"] for e in self.entities if e["label"] in ADDITIVE_LABELS]
        fresh = [n for n in dict.fromkeys(names) if n and n.lower() not in discussed]
        if fresh and len(out) < 3:
            out.append(f"What is {fresh[0]}?")
        subgroups = self.result.get("summary", {}).get("subgroups", {})
        notable = {"culture": "Which ingredients are probiotics?", "protein": "What are the protein sources?",
                   "stimulant": "Is there caffeine?", "fibre": "Which ingredients are fibres?",
                   "enzyme": "Which ingredients are enzymes?"}
        ask = next((q for k, q in notable.items() if k in subgroups), None)
        if ask and intent != "list_category":
            out.append(ask)
        hidden = self.result.get("summary", {}).get("hidden", {})
        if (hidden.get("sugars_under_other_names") or hidden.get("sugar_like_carbohydrates")) and intent != "list_category":
            out.append("Which ingredients are hidden sugars?")
        if self.entities and intent != "diet":
            out.append("Is this product vegan?")
        if self.entities and intent != "allergens":
            out.append("What allergens does it contain?")
        if self.guide.get("rows") and intent != "portion":
            out.append("How much sugar is in my portion?")
        asked = (reply or {}).get("question", "").lower()
        return [q for q in dict.fromkeys(out) if q.lower() != asked][:3]
