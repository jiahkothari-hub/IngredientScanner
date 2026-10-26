"""Sub-groups for the generic "Other ingredients" (entities the NER labels INGREDIENT).

    from src.kb.ingredient_subgroups import subgroup_of
    subgroup_of("Chicory Root Fiber")          -> ("fibre", "rule")
    subgroup_of("Bacillus coagulans")          -> ("culture", "rule")
    subgroup_of("spirulina")                   -> ("fruit_veg", "embedding") or ("other", "")

The NER schema has 10 classes chosen for the app's main question (sugars, fats, additives ...), so
everything else is one INGREDIENT class: fine for "wheat flour", too coarse for a protein powder whose
list is mostly protein isolates, fibres, cultures and enzymes. The NER model is not retrained; the
INGREDIENT entities are sorted afterwards:

  1. rules   ordered keyword patterns on the ingredient name. Order matters: "whey protein" is a protein
             source before it is dairy, "hydrolysed groundnut protein" is a seasoning, "caffeine from
             coffee powder" is a stimulant.
  2. embedding  names no rule covers: the ingredient2vec vectors (src/embeddings/ingredient2vec.py) give
             the ingredient's nearest neighbours; the sub-group most of the close, rule-labelled
             neighbours belong to is used (k-nearest neighbours, similarity-weighted vote). Evaluated in
             main() with held-out rule-labelled terms.
  3. otherwise "other".

    python -m src.kb.ingredient_subgroups      -> evaluation of the embedding fallback (reports/ingredient_subgroups.md)
"""
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache

from src.utils.config import project_path

# key, title, icon, short description, pattern (first matching pattern wins)
SUBGROUPS = [
    ("stimulant", "Caffeine & stimulants", "☕", "Substances with a stimulating effect, such as caffeine.",
     r"caffein|guarana|yerba\s+mate|kola\s+nut|theobromine"),
    ("culture", "Probiotics & cultures", "🧫", "Live or inactive micro-organisms (bacteria, yeasts) added as "
     "probiotics or for fermentation.",
     r"probiotic|bacillus|lactobacill|bifidobacter|streptococcus|saccharomyces|lactococcus|\bcultures?\b|"
     r"\bkefir\b"),
    ("enzyme", "Enzymes", "🧪", "Proteins that speed up chemical reactions, e.g. to help digestion or process "
     "the food.", r"enzym|amylase|protease|lipase|lactase|cellulase|bromelain|papain|pepsin|rennet|"
                  r"transglutaminase"),
    # hydrolysed proteins are savoury flavour bases, not protein sources: checked before "protein"
    ("seasoning", "Yeast, extracts & seasonings", "🫙", "Savoury bases and flavour sources: yeast and yeast "
     "extracts, hydrolysed proteins, stock, sauces and vinegar.", r"hydroly[sz]ed"),
    ("protein", "Protein sources", "💪", "Ingredients added mainly for their protein.",
     r"protein|collagen|\bgelatin"),
    ("seasoning", "Yeast, extracts & seasonings", "🫙", "Savoury bases and flavour sources: yeast and yeast "
     "extracts, hydrolysed proteins, stock, sauces and vinegar.",
     r"\byeast\b|(?:yeast|malt|meat|beef|chicken)\s+extract|seasoning|\bstock\b|broth|bouillon|soy\s+sauce|"
     r"vinegar|\bsauce\b"),
    ("fibre", "Fibres", "🌱", "Dietary fibres: carbohydrates the body does not digest, e.g. inulin or bran.",
     r"fib(?:re|er)|inulin|oligofructose|\bfos\b|psyllium|\bbran\b|resistant\s+(?:dextrin|starch)|polydextrose"),
    ("vitamin_mineral", "Vitamins & minerals", "💊", "Added vitamins and minerals (fortification).",
     r"vitamin|niacin|thiamin|riboflavin|folic|folate|pantothen|biotin|cobalamin|pyridoxine|\bb\d{1,2}\b|"
     r"calcium|\biron\b|zinc|magnesium|iodine|selenium|mineral|ferrous|ferric"),
    ("cocoa_coffee", "Cocoa, coffee & tea", "🍫", "Cocoa, chocolate, coffee and tea ingredients.",
     r"cocoa|cacao|chocolate|coffee|\btea\b|matcha"),
    ("dairy", "Dairy", "🥛", "Milk and ingredients made from milk.",
     r"\bmilk|whey|cream|butter|cheese|yog(?:h)?urt|\bcurd|paneer|\bghee|casein|lactose|khoa|khoya"),
    ("egg", "Egg", "🥚", "Egg and egg products.", r"\beggs?\b|albumen"),
    ("meat_fish", "Meat & fish", "🍗", "Meat, poultry, fish and seafood.",
     r"chicken|beef|pork|\bmeat|\bfish|tuna|salmon|prawn|shrimp|anchov|mutton|lamb"),
    ("legume", "Legumes & soy", "🫘", "Pulses and soy: beans, peas, lentils, chickpeas.",
     r"\bgram\b|chickpea|chana|lentil|\bdal\b|\bbeans?\b|\bpeas?\b|\bsoy|soya"),
    ("nut_seed", "Nuts & seeds", "🌰", "Nuts and seeds.",
     r"\bnuts?\b|almond|hazelnut|cashew|walnut|pistachio|peanut|groundnut|pecan|macadamia|sesame|\bseeds?\b|"
     r"\bchia\b|flax|linseed"),
    ("grain", "Grains, flours & starches", "🌾", "Cereals and what is made from them: flours, meals, starches, "
     "malt.", r"flour|wheat|maida|atta|semolina|suji|sooji|rava|\brice\b|\boats?\b|\bcorn\b|maize|barley|\brye\b|"
              r"millet|ragi|bajra|jowar|starch|\bmeal\b|noodle|pasta|cereal|\bmalt|gluten|quinoa"),
    ("spice_herb", "Spices & herbs", "🌶️", "Spices, herbs and spice mixes.",
     r"spice|chil+i|pepper|turmeric|cumin|coriander|ginger|cinnamon|clove|cardamom|fennel|anise|nutmeg|mace|"
     r"fenugreek|methi|mustard|\bherbs?\b|\bmint|oregano|basil|thyme|rosemary|parsley|paprika|saffron|asafoetida|"
     r"\bhing\b|masala|condiment|curry|ajwain|kalonji|jeera|haldi|dhania|elaichi"),
    ("fruit_veg", "Fruit & vegetables", "🍎", "Fruit, vegetables and products made from them.",
     r"fruit|juice|puree|pur[ée]e|pulp|apple|orange|lemon|\blime\b|berr(?:y|ies)|tomato|potato|onion|garlic|"
     r"carrot|beet|spinach|vegetable|\bveg\b|mango|banana|pineapple|grape|\bdates?\b|raisin|coconut|pumpkin|"
     r"cucumber|radish"),
    ("salt", "Salt", "🧂", "Salt (sodium chloride).", r"\bsalt\b|sodium\s+chloride"),
    ("water", "Water", "💧", "Water.", r"\bwater\b|\bice\b"),
]
OTHER = ("other", "Other", "🥣", "Food ingredients without a more specific group.")
INFO = {key: {"title": title, "icon": icon, "description": desc} for key, title, icon, desc, _ in SUBGROUPS}
INFO[OTHER[0]] = {"title": OTHER[1], "icon": OTHER[2], "description": OTHER[3]}
ORDER = list(dict.fromkeys(key for key, *_ in SUBGROUPS)) + [OTHER[0]]
_RULES = [(key, re.compile(pattern, re.IGNORECASE)) for key, _, _, _, pattern in SUBGROUPS]
# min_share 0.6: of 9 settings compared in the report, the best balance of accuracy (83%) and coverage (51%);
# a wrong group is worse than "Other", so precision was preferred over coverage
K, MIN_SIMILARITY, MIN_SHARE = 10, 0.45, 0.6
GRID = [(s, m) for s in (0.45, 0.5, 0.55) for m in (0.5, 0.6, 0.7)]
REPORT = project_path("reports/ingredient_subgroups")


def rule_subgroup(name: str) -> str:
    for key, pattern in _RULES:
        if pattern.search(name):
            return key
    return ""


@lru_cache(maxsize=1)
def _vectors():
    """ingredient2vec word vectors, or None if the model (or gensim) is not available."""
    try:
        from src.embeddings.ingredient2vec import MODEL_PATH, load_model
        return load_model().wv if MODEL_PATH.exists() else None
    except ImportError:
        return None


def embedding_subgroup(token: str, wv, exclude: str = "") -> str:
    """Similarity-weighted vote of the K nearest rule-labelled neighbours; '' if not confident."""
    if wv is None or token not in wv:
        return ""
    votes, total, used = defaultdict(float), 0.0, 0
    for neighbour, sim in wv.most_similar(token, topn=40):     # look further than K: some neighbours have no rule
        if sim < MIN_SIMILARITY or used >= K:
            break
        group = rule_subgroup(neighbour.replace("_", " ")) if neighbour != exclude else ""
        if group:
            votes[group] += sim
            total += sim
            used += 1
    if not votes:
        return ""
    group, score = max(votes.items(), key=lambda kv: kv[1])
    return group if score / total >= MIN_SHARE else ""


def subgroup_of(name: str) -> tuple:
    """(sub-group key, method): method is 'rule', 'embedding' or '' (other)."""
    group = rule_subgroup(name)
    if group:
        return group, "rule"
    from src.embeddings.ingredient2vec import term_token
    wv = _vectors()
    token = term_token(name)
    candidates = [token] + ([token.split("_")[-1]] if "_" in token else [])   # the phrase, then its head word
    for candidate in candidates:
        group = embedding_subgroup(candidate, wv)
        if group:
            return group, "embedding"
    return OTHER[0], ""


def add_subgroups(items: list) -> list:
    """Give every INGREDIENT entity a 'subgroup' and 'subgroup_method' (in place); returns the items."""
    for it in items:
        if it["label"] == "INGREDIENT":
            it["subgroup"], it["subgroup_method"] = subgroup_of(it["text"])
    return items


# ------------------------------------------------------------------------------------------ evaluation
def evaluate() -> dict:
    """How well does the embedding fallback recover the sub-group of terms the rules know?
    Every rule-labelled vocabulary term is hidden in turn and predicted from its neighbours alone."""
    wv = _vectors()
    labelled = {t: rule_subgroup(t.replace("_", " ")) for t in wv.index_to_key}
    labelled = {t: g for t, g in labelled.items() if g}
    gold, pred = [], []
    for token, group in labelled.items():
        gold.append(group)
        pred.append(embedding_subgroup(token, wv, exclude=token))
    answered = [(g, p) for g, p in zip(gold, pred) if p]
    majority = Counter(gold).most_common(1)[0]
    per_group = defaultdict(lambda: [0, 0])
    for g, p in answered:
        per_group[g][0] += g == p
        per_group[g][1] += 1
    return {"vocabulary": len(wv), "rule_labelled_terms": len(labelled),
            "coverage": round(len(answered) / len(gold), 3),
            "accuracy_when_answered": round(sum(g == p for g, p in answered) / max(1, len(answered)), 3),
            "majority_baseline": {"group": majority[0], "accuracy": round(majority[1] / len(gold), 3)},
            "per_group_accuracy": {g: f"{c}/{n}" for g, (c, n) in sorted(per_group.items())},
            "settings": {"k_candidates": 40, "min_similarity": MIN_SIMILARITY, "min_share": MIN_SHARE},
            "unknown_examples": _unknown_examples(wv, labelled)}


def _unknown_examples(wv, labelled: dict, n: int = 15) -> list:
    """Frequent vocabulary terms no rule covers and that are generic ingredients (not a sugar, fat,
    additive ... the knowledge base knows: those never reach this step), with the suggested sub-group."""
    from src.labeling.weak_labeler import label_text
    out = []
    for token in wv.index_to_key:
        name = token.replace("_", " ")
        if token in labelled or len(name) < 3:
            continue
        labels = {e["label"] for e in label_text(name)["entities"]}
        if labels != {"INGREDIENT"}:               # the app would call it a fat, flavouring, additive ...
            continue
        group = embedding_subgroup(token, wv)
        if group:
            out.append((token.replace("_", " "), group))
        if len(out) >= n:
            break
    return out


def main():
    global MIN_SIMILARITY, MIN_SHARE
    chosen = (MIN_SIMILARITY, MIN_SHARE)
    grid = []
    for MIN_SIMILARITY, MIN_SHARE in GRID:
        r = evaluate()
        grid.append((MIN_SIMILARITY, MIN_SHARE, r["coverage"], r["accuracy_when_answered"]))
    MIN_SIMILARITY, MIN_SHARE = chosen
    result = evaluate()
    result["grid"] = grid
    lines = ["# Ingredient sub-groups: embedding fallback (auto-generated by `python -m src.kb.ingredient_subgroups`)", "",
             f"ingredient2vec vocabulary: {result['vocabulary']:,} terms; the rules give a sub-group to "
             f"{result['rule_labelled_terms']:,}. Each of those is hidden in turn and predicted from its nearest "
             f"rule-labelled neighbours only.", "",
             "| Measure | Value |", "|---|---:|",
             f"| Answered (confident enough) | {result['coverage']:.1%} |",
             f"| Accuracy when answered | **{result['accuracy_when_answered']:.1%}** |",
             f"| Majority-group baseline ({result['majority_baseline']['group']}) | "
             f"{result['majority_baseline']['accuracy']:.1%} |", "",
             "Per sub-group (correct / answered): " + ", ".join(f"{g} {v}" for g, v in result["per_group_accuracy"].items()),
             "", "Confidence settings compared (chosen on this same evaluation, so all are shown):", "",
             "| Min. similarity | Min. share of votes | Answered | Accuracy when answered |", "|---:|---:|---:|---:|"]
    lines += [f"| {s} | {m} | {c:.1%} | {a:.1%} |" + (" ← used" if (s, m) == chosen else "")
              for s, m, c, a in grid]
    lines += ["", "Terms no rule covers, with the sub-group the embeddings suggest:", ""]
    lines += [f"* {t} -> {g}" for t, g in result["unknown_examples"]]
    REPORT.parent.joinpath(f"{REPORT.name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    REPORT.parent.joinpath(f"{REPORT.name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"answered {result['coverage']:.1%}, accuracy when answered {result['accuracy_when_answered']:.1%} "
          f"(majority baseline {result['majority_baseline']['accuracy']:.1%})")
    print("per group:", result["per_group_accuracy"])
    print("unknown terms:", result["unknown_examples"])


if __name__ == "__main__":
    main()
