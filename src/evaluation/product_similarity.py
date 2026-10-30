"""Which text representation finds the most similar product? (Bag of Words, TF-IDF, Jaccard, word2vec)

The app estimates a product's nutrition when the photo shows only the ingredient list: it looks up the
database product with the most similar ingredient list (src/app/nutrition.py, TF-IDF of words and
word pairs + cosine similarity). This module compares that choice with the other classic
representations and similarity measures on a task with a measurable answer:

    query     every TEST-split product with nutrition facts (never in the database)
    database  TRAIN + VALIDATION products with nutrition facts
    predict   the nutrition of the most similar database product (top-1, as in the app) and the mean
              of the 5 most similar (top-5)
    score     absolute error in sugars, fat and salt (g/100 g) and energy (kcal/100 g), and how often
              the neighbour has the same main category

Representations x similarity:
    Jaccard              set of words, |A & B| / |A | B|
    BoW + Euclidean      word counts, straight-line distance
    BoW + Manhattan      word counts, sum of absolute differences
    BoW + cosine         word counts, angle between vectors
    TF-IDF 1-gram + cos  words weighted by rarity
    TF-IDF 1,2-gram+cos  words and word pairs (the app's current method)
    ingredient2vec       mean of the ingredient embeddings (src/embeddings/ingredient2vec.py), cosine
    TF-IDF + i2v         both, side by side (each L2-normalised)
plus two baselines: the database mean (no similarity at all) and a random database product.

Crowd-sourced values outside physical limits (sugars > 100 g/100 g, energy > 900 kcal/100 g, ...)
are removed. Mean and median absolute errors are reported (the median is robust to remaining outliers).

    python -m src.evaluation.product_similarity     -> reports/product_similarity.md / .json
"""
import json

import numpy as np
import pandas as pd
from scipy import sparse

from src.utils.config import project_path

TARGETS = {"sugars_100g": "sugars_g", "fat_100g": "fat_g", "salt_100g": "salt_g", "energy-kcal_100g": "energy_kcal"}
LIMITS = {"sugars_100g": 100, "fat_100g": 100, "salt_100g": 100, "energy-kcal_100g": 900}
REPORT = project_path("reports/product_similarity")
CHUNK = 256


def load_data() -> tuple:
    nutrition = pd.read_csv(project_path("data/knowledge_base/products_nutrition.csv"), dtype={"product_id": str})
    texts = pd.read_csv(project_path("data/processed/products.csv"), dtype={"product_id": str},
                        usecols=["product_id", "ingredients_text_raw"])
    splits = pd.read_csv(project_path("data/processed/split_assignment.csv"), dtype={"product_id": str},
                         usecols=["product_id", "split"])
    data = nutrition.merge(texts, on="product_id").merge(splits, on="product_id")
    data = data.dropna(subset=list(TARGETS) + ["ingredients_text_raw"])
    for col, limit in LIMITS.items():
        data = data[(data[col] >= 0) & (data[col] <= limit)]
    data["text"] = data["ingredients_text_raw"].str.lower()
    data["category"] = data["main_category"].where(~data["main_category"].isin(["Undefined", "I-want-to-know"]))
    database = data[data["split"].isin(["train", "validation"])].reset_index(drop=True)
    queries = data[data["split"] == "test"].reset_index(drop=True)
    return database, queries


# ------------------------------------------------------------------------------ similarity functions
# each returns a (n_queries x n_database) matrix where LARGER = more similar
def cosine(q, d):
    from sklearn.preprocessing import normalize
    return np.asarray((normalize(q) @ normalize(d).T).todense() if sparse.issparse(q) else normalize(q) @ normalize(d).T)


def jaccard(q, d):
    q, d = (q > 0).astype(np.float32), (d > 0).astype(np.float32)
    inter = np.asarray((q @ d.T).todense())
    union = np.asarray(q.sum(1)) + np.asarray(d.sum(1)).T - inter
    return inter / np.maximum(union, 1)


def negative_distance(metric):
    def score(q, d):
        from sklearn.metrics import pairwise_distances
        return -pairwise_distances(q, d, metric=metric)
    return score


def representations(database: pd.DataFrame, queries: pd.DataFrame) -> dict:
    """name -> (query matrix, database matrix, similarity function)."""
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.preprocessing import normalize
    bow = CountVectorizer(min_df=2).fit(database["text"])
    tf1 = TfidfVectorizer(min_df=2, sublinear_tf=True).fit(database["text"])
    tf12 = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(database["text"])   # = the app
    Bq, Bd = bow.transform(queries["text"]), bow.transform(database["text"])
    reps = {
        "Jaccard (word sets)": (Bq, Bd, jaccard),
        "BoW + Euclidean": (Bq, Bd, negative_distance("euclidean")),
        "BoW + Manhattan": (Bq, Bd, negative_distance("manhattan")),
        "BoW + cosine": (Bq, Bd, cosine),
        "TF-IDF words + cosine": (tf1.transform(queries["text"]), tf1.transform(database["text"]), cosine),
        "TF-IDF words+pairs + cosine (app)": (tf12.transform(queries["text"]), tf12.transform(database["text"]), cosine),
    }
    i2v = ingredient_vectors(database, queries)
    if i2v is not None:
        Eq, Ed = i2v
        reps["ingredient2vec mean + cosine"] = (Eq, Ed, cosine)
        Tq, Td = reps["TF-IDF words+pairs + cosine (app)"][:2]
        reps["TF-IDF + ingredient2vec + cosine"] = (sparse.hstack([normalize(Tq), sparse.csr_matrix(Eq)]).tocsr(),
                                                    sparse.hstack([normalize(Td), sparse.csr_matrix(Ed)]).tocsr(), cosine)
    return reps


def ingredient_vectors(database: pd.DataFrame, queries: pd.DataFrame):
    """Mean ingredient2vec vector of each product (entities from the silver records); None if no model."""
    from sklearn.preprocessing import normalize
    from src.embeddings.ingredient2vec import MODEL_PATH, document_vector, load_model, record_tokens
    from src.ner.data import load_split
    if not MODEL_PATH.exists():
        return None
    wv = load_model().wv
    tokens = {r["product_id"]: record_tokens(r) for split in ("train", "validation", "test")
              for r in load_split("silver", split)}
    def matrix(frame):
        return normalize(np.array([document_vector(tokens.get(pid, []), wv) for pid in frame["product_id"]]))
    return matrix(queries), matrix(database)


# ------------------------------------------------------------------------------------- scoring
def neighbours(q, d, score, k: int = 5) -> np.ndarray:
    """Indices of the k most similar database rows for every query (computed in chunks)."""
    out = []
    for start in range(0, q.shape[0], CHUNK):
        s = score(q[start:start + CHUNK], d)
        out.append(np.argsort(-s, axis=1, kind="stable")[:, :k])
    return np.vstack(out)


def errors(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    out = {}
    for col, name in TARGETS.items():
        err = (pred[col].to_numpy() - truth[col].to_numpy())
        out[name] = {"mae": round(float(np.abs(err).mean()), 2), "median_ae": round(float(np.median(np.abs(err))), 2)}
    return out


def same_category(database, queries, idx) -> float:
    known = queries["category"].notna().to_numpy() & database["category"].iloc[idx].notna().to_numpy()
    same = queries["category"].to_numpy()[known] == database["category"].iloc[idx].to_numpy()[known]
    return round(float(same.mean()), 3)


def evaluate(database: pd.DataFrame, queries: pd.DataFrame) -> dict:
    results = {}
    mean_pred = pd.DataFrame({c: np.full(len(queries), database[c].mean()) for c in TARGETS})
    results["Baseline: database mean"] = {"top1": errors(mean_pred, queries), "top5": None, "same_category_top1": None}
    rng = np.random.default_rng(42)
    rand = rng.integers(0, len(database), len(queries))
    results["Baseline: random product"] = {"top1": errors(database.iloc[rand].reset_index(drop=True), queries),
                                           "top5": None, "same_category_top1": same_category(database, queries, rand)}
    for name, (q, d, score) in representations(database, queries).items():
        idx = neighbours(q, d, score)
        top1 = database.iloc[idx[:, 0]].reset_index(drop=True)
        top5 = pd.DataFrame({c: database[c].to_numpy()[idx].mean(1) for c in TARGETS})
        results[name] = {"top1": errors(top1, queries), "top5": errors(top5, queries),
                         "same_category_top1": same_category(database, queries, idx[:, 0])}
        print(f"{name:36} top-1 sugars MAE {results[name]['top1']['sugars_g']['mae']:5.2f}  "
              f"fat {results[name]['top1']['fat_g']['mae']:5.2f}  salt {results[name]['top1']['salt_g']['mae']:5.2f}  "
              f"kcal {results[name]['top1']['energy_kcal']['mae']:6.1f}  same category "
              f"{results[name]['same_category_top1']}", flush=True)
    return results


def write_report(results: dict, n_db: int, n_q: int):
    lines = ["# Text representations for finding a similar product (auto-generated by "
             "`python -m src.evaluation.product_similarity`)", "",
             f"Queries: {n_q:,} test products; database: {n_db:,} train/validation products (all with nutrition "
             "facts within physical limits). Prediction = nutrition of the most similar product (top-1) or the "
             "mean of the 5 most similar (top-5). Lower error is better.", "",
             "| Representation | Sugars MAE | Fat MAE | Salt MAE | kcal MAE | Sugars MAE (top-5) | kcal MAE (top-5) | Same category |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in results.items():
        t1, t5 = r["top1"], r["top5"]
        cells = [f"{t1[k]['mae']:.2f}" for k in ("sugars_g", "fat_g", "salt_g")] + [f"{t1['energy_kcal']['mae']:.1f}"]
        cells += [f"{t5['sugars_g']['mae']:.2f}", f"{t5['energy_kcal']['mae']:.1f}"] if t5 else ["-", "-"]
        cells += [f"{r['same_category_top1']:.1%}" if r["same_category_top1"] is not None else "-"]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "Median absolute errors (top-1), robust to outliers:", "",
              "| Representation | Sugars | Fat | Salt | kcal |", "|---|---:|---:|---:|---:|"]
    for name, r in results.items():
        t1 = r["top1"]
        lines.append(f"| {name} | " + " | ".join(f"{t1[k]['median_ae']:.2f}" for k in ("sugars_g", "fat_g", "salt_g"))
                     + f" | {t1['energy_kcal']['median_ae']:.1f} |")
    (REPORT.parent / f"{REPORT.name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REPORT.parent / f"{REPORT.name}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def main():
    database, queries = load_data()
    print(f"database {len(database):,} products, queries {len(queries):,} test products", flush=True)
    results = evaluate(database, queries)
    write_report(results, len(database), len(queries))
    print(f"saved {REPORT}.md")


if __name__ == "__main__":
    main()
