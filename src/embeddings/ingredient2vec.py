"""ingredient2vec: word2vec embeddings of INGREDIENTS, learned from the training ingredient lists.

Distributional hypothesis: ingredients that appear in similar contexts have similar roles. "dextrose"
and "glucose syrup" are listed next to the same kinds of things (flour, oils, flavourings), so their
vectors end up close together, even though the two names share no letters.

    python -m src.embeddings.ingredient2vec          # train + evaluate + figure (~1-2 min on CPU)

    from src.embeddings.ingredient2vec import load_model, most_similar
    most_similar("dextrose")      -> [("glucose_syrup", 0.81), ...]

Tokens. Each recognised ingredient (an entity of the silver labels, in reading order) is ONE token:
"glucose syrup" -> "glucose_syrup". A product's ingredient list is one "sentence". Only the TRAIN
split is used, so validation/test products stay unseen.

Evaluation (reports/embeddings/ingredient2vec.md):
  1. Nearest neighbours of common ingredients (qualitative).
  2. Term classification: can the vectors tell sugars from fats from colours...? Ingredient terms whose
     category the rules know confidently are classified by k-nearest neighbours (5-fold cross-
     validation) using (a) the majority class, (b) the term's SPELLING (character 3-gram TF-IDF),
     (c) its ingredient2vec vector and (d) spelling and vector side by side. The vectors never saw the
     labels: (c) measures how much category information the contexts alone carry, (d) whether it adds
     to what the spelling already tells. Six configurations (window x CBOW/skip-gram) are compared.
  3. Discovery: frequent terms the rules could only label as generic INGREDIENT, ranked by how close
     they are to the sugar / fat / sweetener centroids - candidates for "hidden" ingredients to review.
  4. A t-SNE map of the labelled terms (reports/figures/14_ingredient2vec_tsne.png).
"""
import json
from collections import Counter, defaultdict

import numpy as np

from src.labeling.lexicon import lookup_key
from src.ner.data import load_split
from src.utils.config import project_path

MODEL_PATH = project_path("models/embeddings/ingredient2vec.model")
REPORT_DIR = project_path("reports/embeddings")
FIGURE_PATH = project_path("reports/figures/14_ingredient2vec_tsne.png")
CLASSES = ["SUGAR", "SWEETENER", "FAT", "PRESERVATIVE", "COLOUR", "ADDITIVE", "FLAVOURING", "FUNCTION_CLASS"]
# CBOW, window 3: best of the 6 configurations compared in the report (window 2/3/5 x CBOW/skip-gram)
PARAMS = {"vector_size": 100, "window": 3, "min_count": 5, "sg": 0, "negative": 10, "epochs": 50,
          "seed": 42, "workers": 1}                      # 1 worker: reproducible vectors
GRID = [(w, sg) for w in (2, 3, 5) for sg in (0, 1)]
QUERIES = ["sugar", "dextrose", "palm_oil", "citric_acid", "sucralose", "sodium_benzoate", "caramel_colour",
           "soya_lecithin", "salt", "wheat_flour"]


def term_token(text: str) -> str:
    """'Glucose  Syrup' -> 'glucose_syrup' (same normalisation as the dictionaries, spaces -> '_')."""
    return lookup_key(text).replace(" ", "_")


def record_tokens(record: dict) -> list:
    """The ingredient tokens of one silver record, in reading order (INS codes are left out)."""
    entities = sorted(record["entities"], key=lambda e: e["start"])
    return [term_token(e["text"]) for e in entities if e["label"] != "INS_CODE" and e["text"].strip()]


def train(records: list, **overrides):
    from gensim.models import Word2Vec
    sentences = [t for t in (record_tokens(r) for r in records) if len(t) > 1]
    return Word2Vec(sentences=sentences, **{**PARAMS, **overrides})


def load_model():
    from gensim.models import Word2Vec
    return Word2Vec.load(str(MODEL_PATH))


def most_similar(term: str, topn: int = 8, model=None) -> list:
    model = model or load_model()
    token = term_token(term)
    if token not in model.wv:
        return []
    return [(t, round(float(s), 3)) for t, s in model.wv.most_similar(token, topn=topn)]


def document_vector(tokens: list, wv) -> np.ndarray:
    """Mean of the known ingredient vectors (zero vector if none is known)."""
    vectors = [wv[t] for t in tokens if t in wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(wv.vector_size, dtype=np.float32)


# ------------------------------------------------------------------------------------ evaluation
def term_labels(records: list, min_share: float = 0.8) -> tuple:
    """term -> category from rule-made (not fallback) labels, kept only if one category dominates;
    plus the terms the rules could only label as generic INGREDIENT (fallback)."""
    labels, fallback = defaultdict(Counter), Counter()
    for r in records:
        for e in r["entities"]:
            token = term_token(e["text"])
            if e.get("rule", "").startswith("fallback"):
                fallback[token] += 1
            elif e["label"] in CLASSES:
                labels[token][e["label"]] += 1
    confident = {}
    for token, counts in labels.items():
        label, n = counts.most_common(1)[0]
        if n / sum(counts.values()) >= min_share:
            confident[token] = label
    return confident, fallback


def knn_cv(X: np.ndarray, y: np.ndarray, k: int = 5, folds: int = 5, seed: int = 42, return_pred: bool = False):
    """Cosine kNN (distance-weighted), stratified k-fold. Returns accuracy, macro F1 and per-class F1."""
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")
    pred = cross_val_predict(clf, X, y, cv=StratifiedKFold(folds, shuffle=True, random_state=seed))
    labels = sorted(set(y))
    per_class = f1_score(y, pred, labels=labels, average=None, zero_division=0)
    scores = {"accuracy": round(accuracy_score(y, pred), 3),
              "macro_f1": round(f1_score(y, pred, average="macro", zero_division=0), 3),
              "per_class_f1": {l: round(float(f), 3) for l, f in zip(labels, per_class)}}
    return (scores, pred) if return_pred else scores


def term_matrices(wv, confident: dict) -> tuple:
    """Labelled terms (classes with >= 10 terms, for 5-fold CV) and their three representations,
    each row L2-normalised: context (word2vec), spelling (char 3-gram TF-IDF) and both side by side."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    terms = sorted(t for t in confident if t in wv)
    counts = Counter(confident[t] for t in terms)
    keep = [t for t in terms if counts[confident[t]] >= 10]
    y = np.array([confident[t] for t in keep])
    emb = normalize(np.array([wv[t] for t in keep]))
    chars = normalize(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3))
                      .fit_transform([t.replace("_", " ") for t in keep]).toarray())
    return keep, y, emb, chars, np.hstack([emb, chars])


def evaluate(model, records: list) -> dict:
    wv = model.wv
    confident, fallback = term_labels(records)
    keep, y, emb, chars, both = term_matrices(wv, confident)
    majority = Counter(y).most_common(1)[0]
    spelling, pred_spelling = knn_cv(chars, y, return_pred=True)
    combined, pred_both = knn_cv(both, y, return_pred=True)
    fixed = [(keep[i], pred_spelling[i], y[i]) for i in range(len(y)) if pred_spelling[i] != y[i] == pred_both[i]]
    result = {
        "params": PARAMS, "vocabulary": len(wv), "labelled_terms": len(keep),
        "class_counts": dict(Counter(y)),
        "classification": {
            "majority_class": {"accuracy": round(majority[1] / len(y), 3), "predicts": majority[0]},
            "spelling_char3gram_knn": spelling,
            "ingredient2vec_knn": knn_cv(emb, y),
            "spelling_plus_ingredient2vec_knn": combined,
        },
        "context_fixes_spelling": [{"term": t, "spelling_said": s, "true": g} for t, s, g in fixed[:20]],
        "n_context_fixes": len(fixed),
        "n_context_breaks": int(sum(pred_spelling[i] == y[i] != pred_both[i] for i in range(len(y)))),
        "neighbours": {q: most_similar(q, 8, model) for q in QUERIES if q in wv},
    }
    # discovery: frequent generic ingredients that sit close to a category centroid
    centroids = {c: emb[y == c].mean(0) for c in ("SUGAR", "FAT", "SWEETENER") if (y == c).any()}
    candidates = {}
    generic = [t for t, n in fallback.items() if n >= 10 and t in wv and t not in confident]
    for c, centre in centroids.items():
        centre = centre / np.linalg.norm(centre)
        scored = sorted(((float(wv[t] @ centre / np.linalg.norm(wv[t])), t) for t in generic), reverse=True)
        candidates[c] = [(t, round(s, 3)) for s, t in scored[:10]]
    result["discovery_candidates"] = candidates
    result["_tsne"] = (keep, emb, y)
    return result


def grid_search(records: list) -> list:
    """Context window x architecture, all scored with the same term classification."""
    confident, _ = term_labels(records)
    rows = []
    for window, sg in GRID:
        model = train(records, window=window, sg=sg)
        _, y, emb, _, both = term_matrices(model.wv, confident)
        e, b = knn_cv(emb, y), knn_cv(both, y)
        rows.append({"window": window, "architecture": "skip-gram" if sg else "CBOW",
                     "w2v_accuracy": e["accuracy"], "w2v_macro_f1": e["macro_f1"],
                     "combined_accuracy": b["accuracy"], "combined_macro_f1": b["macro_f1"]})
        print(f"  grid window {window} {rows[-1]['architecture']}: w2v {e['accuracy']:.3f}, combined {b['accuracy']:.3f}",
              flush=True)
    return rows


def plot_tsne(terms: list, emb: np.ndarray, y: np.ndarray, frequencies: Counter, max_terms: int = 700):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    order = sorted(range(len(terms)), key=lambda i: -frequencies[terms[i]])[:max_terms]
    points = TSNE(n_components=2, metric="cosine", init="pca", perplexity=30, random_state=42).fit_transform(emb[order])
    colours = dict(zip(CLASSES, ["#e8590c", "#d6336c", "#f2b705", "#c92a2a", "#7048e8", "#1971c2", "#2f9e44", "#495057"]))
    fig, ax = plt.subplots(figsize=(7, 5.2))
    for c in CLASSES:
        idx = [k for k, i in enumerate(order) if y[i] == c]
        if idx:
            ax.scatter(points[idx, 0], points[idx, 1], s=9, color=colours[c], label=c.replace("_", " ").lower(), alpha=0.8)
    for k, i in enumerate(order[:25]):                                  # label the most frequent terms
        ax.annotate(terms[i].replace("_", " "), points[k], fontsize=6.5, alpha=0.85)
    ax.set_xticks([]); ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=7, markerscale=1.8, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    ax.set_title("ingredient2vec: ingredients that share contexts cluster by role (t-SNE)", loc="left", fontsize=9)
    fig.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_report(result: dict):
    c = result["classification"]
    lines = ["# ingredient2vec (auto-generated by `python -m src.embeddings.ingredient2vec`)", "",
             f"word2vec skip-gram trained on {result['sentences']:,} training ingredient lists; "
             f"{result['vocabulary']:,} ingredient tokens (min count {PARAMS['min_count']}).", "",
             "## Term classification (5-fold CV, kNN k=5, cosine)", "",
             f"{result['labelled_terms']} terms whose category the rules know confidently: "
             + ", ".join(f"{k} {v}" for k, v in sorted(result["class_counts"].items(), key=lambda kv: -kv[1])), "",
             "| Representation | Accuracy | Macro F1 |", "|---|---:|---:|",
             f"| Majority class ({c['majority_class']['predicts']}) | {c['majority_class']['accuracy']:.3f} | - |",
             f"| Spelling (character 3-gram TF-IDF) | {c['spelling_char3gram_knn']['accuracy']:.3f} | "
             f"{c['spelling_char3gram_knn']['macro_f1']:.3f} |",
             f"| ingredient2vec (context only) | {c['ingredient2vec_knn']['accuracy']:.3f} | "
             f"{c['ingredient2vec_knn']['macro_f1']:.3f} |",
             f"| **Spelling + ingredient2vec** | **{c['spelling_plus_ingredient2vec_knn']['accuracy']:.3f}** | "
             f"**{c['spelling_plus_ingredient2vec_knn']['macro_f1']:.3f}** |", "",
             "Per-class F1 (spelling / context / both): "
             + ", ".join(f"{l} {c['spelling_char3gram_knn']['per_class_f1'][l]:.2f} / "
                         f"{c['ingredient2vec_knn']['per_class_f1'][l]:.2f} / "
                         f"{c['spelling_plus_ingredient2vec_knn']['per_class_f1'][l]:.2f}"
                         for l in c["ingredient2vec_knn"]["per_class_f1"]), "",
             f"Adding the context vectors fixes {result['n_context_fixes']} terms that spelling alone gets wrong "
             f"and breaks {result['n_context_breaks']}. Examples fixed: "
             + ", ".join(f"{x['term'].replace('_', ' ')} ({x['spelling_said'].lower()} -> {x['true'].lower()})"
                         for x in result["context_fixes_spelling"][:10]) + ".", "",
             "## Configurations compared", "",
             "The configuration used above was chosen on this same evaluation, so all six are shown.", "",
             "| Window | Architecture | Context only (acc / macro F1) | Spelling + context (acc / macro F1) |",
             "|---:|---|---|---|"]
    lines += [f"| {g['window']} | {g['architecture']} | {g['w2v_accuracy']:.3f} / {g['w2v_macro_f1']:.3f} | "
              f"{g['combined_accuracy']:.3f} / {g['combined_macro_f1']:.3f} |" for g in result["grid"]]
    lines += ["", "## Nearest neighbours", ""]
    for q, nn in result["neighbours"].items():
        lines.append(f"* **{q.replace('_', ' ')}**: " + ", ".join(f"{t.replace('_', ' ')} ({s:.2f})" for t, s in nn))
    lines += ["", "## Discovery: generic ingredients closest to each category (candidates to review, not labels)", ""]
    for cat, items in result["discovery_candidates"].items():
        lines.append(f"* **{cat}**: " + ", ".join(f"{t.replace('_', ' ')} ({s:.2f})" for t, s in items))
    lines += ["", "![t-SNE](../figures/14_ingredient2vec_tsne.png)"]
    (REPORT_DIR / "ingredient2vec.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    records = load_split("silver", "train")
    model = train(records)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))
    result = evaluate(model, records)
    result["sentences"] = model.corpus_count
    result["grid"] = grid_search(records)
    terms, emb, y = result.pop("_tsne")
    frequencies = Counter(t for r in records for t in record_tokens(r))
    plot_tsne(terms, emb, y, frequencies)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "ingredient2vec.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(result)
    c = result["classification"]
    print(f"vocabulary {result['vocabulary']:,} tokens from {result['sentences']:,} lists; {result['labelled_terms']} labelled terms")
    print(f"term classification accuracy: majority {c['majority_class']['accuracy']:.3f}, "
          f"spelling {c['spelling_char3gram_knn']['accuracy']:.3f} (macro F1 {c['spelling_char3gram_knn']['macro_f1']:.3f}), "
          f"ingredient2vec {c['ingredient2vec_knn']['accuracy']:.3f} (macro F1 {c['ingredient2vec_knn']['macro_f1']:.3f}), "
          f"both {c['spelling_plus_ingredient2vec_knn']['accuracy']:.3f} "
          f"(macro F1 {c['spelling_plus_ingredient2vec_knn']['macro_f1']:.3f})")
    print(f"context fixes {result['n_context_fixes']} spelling errors, breaks {result['n_context_breaks']}: "
          f"{[x['term'] for x in result['context_fixes_spelling'][:8]]}")
    for q in ("dextrose", "palm_oil", "sucralose", "citric_acid"):
        print(f"  {q}: {result['neighbours'].get(q, [])[:5]}")
    for cat, items in result["discovery_candidates"].items():
        print(f"  near {cat}: {[t for t, _ in items[:6]]}")
    print(f"saved {MODEL_PATH}, {REPORT_DIR / 'ingredient2vec.md'}, {FIGURE_PATH}")


if __name__ == "__main__":
    main()
