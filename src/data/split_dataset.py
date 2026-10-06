"""Leakage-free train / validation / test split (70 / 15 / 15), reproducible with a fixed seed.

Usage:  python -m src.data.split_dataset        (after src.labeling.build_silver)

THE PROBLEM: many products are near-copies (same recipe sold under another barcode, size or
store brand). If one copy is in train and another in test, the test score is inflated because the
model has effectively seen the test example.

THE SOLUTION: put similar products into one GROUP, and split whole groups. Two products are in the
same group if they share ANY of these keys (union-find, so groups are transitive):
  A. the same letters after removing everything else    "Sugar, Salt." == "SUGAR SALT"
  B. the same SET of ingredient segments                "salt, sugar"  == "sugar, salt"
  C. the same product name + brand
  D. the same first 5 ingredient segments (same recipe family; lists with >= 5 segments)
  E. near-duplicate ingredient sets: Jaccard similarity >= 0.8 (lists with >= 5 segments),
     e.g. two flavours of the same gummy recipe that differ in one colour
Groups are assigned to splits per market group (india, uk_ireland, ...) so every split has the same
market mix. Afterwards we MEASURE leakage: the highest ingredient-set similarity between each test
product and any train product.

Outputs:
  data/processed/split_assignment.csv        product_id -> group_id -> split
  data/splits/silver/{train,validation,test}.jsonl.gz   (gzip; Dataset.from_json reads it directly)
  data/splits/gold/{train,validation,test}.jsonl        (only if verified gold annotations exist)
  data/splits/label2id.json, data/splits/id2label.json
  reports/split_report.json
"""
import json
import random
import re
from collections import Counter, defaultdict

import pandas as pd

from src.labeling.bio import ID2LABEL, LABEL2ID
from src.utils.config import load_config, project_path
from src.utils.io import read_jsonl, write_jsonl
from src.utils.validation import validate_records

SPLITS = {"train": 0.70, "validation": 0.15, "test": 0.15}
SILVER_PATH = project_path("data/processed/silver.jsonl.gz")
GOLD_PATH = project_path("data/annotations/gold_verified.jsonl")
SPLIT_DIR = project_path("data/splits")


# ---------------------------------------------------------------------------- grouping
class UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # path halving keeps trees flat
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def segment_set(record) -> frozenset:
    """Lower-cased ingredient segments without percentages (used for keys B, D and leakage check)."""
    pieces = re.split(r"[,;:()\[\]{}]|\.\s", record["text"].lower())
    cleaned = (re.sub(r"\d+(?:[.,]\d+)?\s?%|\s+", " ", p).strip(" .*-") for p in pieces)
    return frozenset(p for p in cleaned if p)


def first_segments(record, n=5):
    pieces = [re.sub(r"\d+(?:[.,]\d+)?\s?%|\s+", " ", p).strip(" .*-")
              for p in re.split(r"[,;:()\[\]{}]|\.\s", record["text"].lower())]
    pieces = [p for p in pieces if p]
    return tuple(pieces[:n]) if len(pieces) >= n else None


def near_duplicate_pairs(records, threshold=0.8, min_segments=5, common_df=300):
    """Pairs of records whose ingredient-segment sets have Jaccard >= threshold (key E).
    Candidate pairs come from an inverted index of uncommon segments, so this stays fast."""
    sets = [segment_set(r) for r in records]
    doc_freq = Counter(seg for s in sets for seg in s)
    index = defaultdict(list)
    for i, s in enumerate(sets):
        if len(s) >= min_segments:
            for seg in s:
                if doc_freq[seg] <= common_df:
                    index[seg].append(i)
    pairs = []
    for i, s in enumerate(sets):
        if len(s) < min_segments:
            continue
        candidates = {j for seg in s for j in index.get(seg, ()) if j > i}
        for j in candidates:
            if len(s & sets[j]) / len(s | sets[j]) >= threshold:
                pairs.append((records[i]["id"], records[j]["id"]))
    return pairs


def build_groups(records, products) -> dict:
    """Return product_id -> group_id."""
    names = {row.product_id: f"{row.product_name}|{row.brands}".lower()
             for row in products.itertuples() if isinstance(row.product_name, str) and isinstance(row.brands, str)}
    uf = UnionFind([r["id"] for r in records])
    keys = defaultdict(list)
    for r in records:
        keys["A:" + re.sub(r"[^a-z]", "", r["text"].lower())].append(r["id"])
        keys["B:" + "|".join(sorted(segment_set(r)))].append(r["id"])
        if r["id"] in names and len(names[r["id"]]) > 4:
            keys["C:" + names[r["id"]]].append(r["id"])
        first = first_segments(r)
        if first:
            keys["D:" + "|".join(first)].append(r["id"])
    for members in keys.values():
        for other in members[1:]:
            uf.union(members[0], other)
    for a, b in near_duplicate_pairs(records):
        uf.union(a, b)
    return {r["id"]: uf.find(r["id"]) for r in records}


# ---------------------------------------------------------------------------- assignment
def assign_splits(records, group_of, seed) -> dict:
    """Assign whole groups to splits, separately for each market group (stratification).

    A group can contain products from several markets; it is stratified under its most common
    market, so the group still goes to exactly ONE split.
    """
    members_of = defaultdict(list)
    markets_of = defaultdict(Counter)
    for r in records:
        members_of[group_of[r["id"]]].append(r["id"])
        markets_of[group_of[r["id"]]][r["country_group"]] += 1
    strata = defaultdict(list)                              # market -> [(group_id, members)]
    for gid, members in members_of.items():
        market = sorted(markets_of[gid].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        strata[market].append((gid, members))

    rng = random.Random(seed)
    split_of = {}
    for market in sorted(strata):
        groups = sorted(strata[market])                     # sorted first => deterministic
        rng.shuffle(groups)
        total = sum(len(m) for _, m in groups)
        filled = Counter()
        for _, members in groups:
            # put the group where the split is furthest below its target size
            target = min(SPLITS, key=lambda s: filled[s] / total - SPLITS[s])
            filled[target] += len(members)
            for pid in members:
                split_of[pid] = target
    return split_of


# ---------------------------------------------------------------------------- leakage check
def max_train_similarity(records, split_of) -> dict:
    """For each validation/test record: highest Jaccard similarity of ingredient sets with any train record."""
    train = [segment_set(r) for r in records if split_of[r["id"]] == "train"]
    index = defaultdict(list)
    for i, s in enumerate(train):
        for seg in s:
            index[seg].append(i)
    result = {}
    for r in records:
        if split_of[r["id"]] == "train":
            continue
        s = segment_set(r)
        shared = Counter(i for seg in s for i in index.get(seg, ()))
        best = max((n / len(s | train[i]) for i, n in shared.items()), default=0.0)
        result[r["id"]] = best
    return result


def to_hf_record(record, split) -> dict:
    """Format used by Person 2: tokens + integer ner_tags (+ readable names and metadata)."""
    return {
        "id": record["id"],
        "product_id": record["product_id"],
        "country_group": record["country_group"],
        "split": split,
        "text": record["text"],
        "tokens": record["tokens"],
        "ner_tags": [LABEL2ID[t] for t in record["ner_tags"]],
        "ner_tag_names": record["ner_tags"],
        "token_offsets": record["token_offsets"],
        # fixed keys in every entity, so Arrow / Hugging Face infer one schema
        "entities": [{"start": e["start"], "end": e["end"], "text": e["text"], "label": e["label"],
                      "rule": e.get("rule", "human"), "number": e.get("number", "")} for e in record["entities"]],
        "label_source": record["label_source"],
    }


def write_split_files(records, split_of, folder, compress=True):
    """Silver splits are gzip-compressed (large); gold splits are plain JSONL (small, easy to inspect).
    Both load directly with datasets.Dataset.from_json(path)."""
    counts = {}
    suffix = ".jsonl.gz" if compress else ".jsonl"
    for split in SPLITS:
        part = [to_hf_record(r, split) for r in records if split_of.get(r["id"]) == split]
        counts[split] = write_jsonl(part, SPLIT_DIR / folder / f"{split}{suffix}")
    return counts


def main():
    config = load_config()
    seed = config["random_seed"]
    products = pd.read_csv(project_path(config["outputs"]["products"]), dtype={"product_id": str})
    records = read_jsonl(SILVER_PATH)
    validate_records(records, name="silver")

    group_of = build_groups(records, products)
    split_of = assign_splits(records, group_of, seed)
    group_sizes = Counter(group_of.values())

    assignment = pd.DataFrame({
        "product_id": [r["id"] for r in records],
        "country_group": [r["country_group"] for r in records],
        "group_id": [group_of[r["id"]] for r in records],
        "split": [split_of[r["id"]] for r in records],
    })
    assignment["group_size"] = assignment["group_id"].map(group_sizes)
    assignment.to_csv(project_path("data/processed/split_assignment.csv"), index=False)

    # sanity checks: no group and no identical text in two splits
    groups_per_split = assignment.groupby("group_id")["split"].nunique()
    assert (groups_per_split == 1).all(), "a group was split across splits"
    texts = defaultdict(set)
    for r in records:
        texts[r["text"].lower()].add(split_of[r["id"]])
    assert all(len(s) == 1 for s in texts.values()), "identical text in two splits"

    silver_counts = write_split_files(records, split_of, "silver")
    SPLIT_DIR.joinpath("label2id.json").write_text(json.dumps(LABEL2ID, indent=2), encoding="utf-8")
    SPLIT_DIR.joinpath("id2label.json").write_text(json.dumps(ID2LABEL, indent=2), encoding="utf-8")

    gold_counts = None
    if GOLD_PATH.exists():
        gold = read_jsonl(GOLD_PATH)
        validate_records(gold, name="gold")
        gold_counts = write_split_files(gold, split_of, "gold", compress=False)

    similarity = max_train_similarity(records, split_of)
    sims = pd.Series(similarity)
    report = {
        "seed": seed,
        "records": len(records),
        "groups": len(group_sizes),
        "groups_with_more_than_one_product": sum(1 for n in group_sizes.values() if n > 1),
        "largest_group": max(group_sizes.values()),
        "silver_split_sizes": silver_counts,
        "silver_split_percent": {k: round(100 * v / len(records), 1) for k, v in silver_counts.items()},
        "market_mix_percent": (pd.crosstab(assignment["split"], assignment["country_group"], normalize="index")
                               * 100).round(1).to_dict(orient="index"),
        "gold_split_sizes": gold_counts,
        "leakage_check": {
            "description": "max Jaccard similarity of ingredient-segment sets between each val/test "
                           "product and its most similar train product",
            "median": round(float(sims.median()), 3),
            "share_at_least_0.8": round(float((sims >= 0.8).mean()), 4),
            "share_identical_1.0": round(float((sims >= 1.0).mean()), 4),
        },
    }
    project_path("reports/split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("groups", "silver_split_sizes", "silver_split_percent",
                                              "leakage_check")}, indent=2))


if __name__ == "__main__":
    main()
