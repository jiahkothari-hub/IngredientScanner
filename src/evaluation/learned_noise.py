"""OCR noise model LEARNED from the real packet photos (NAT-style, Namysl et al. 2020).

The hand-made noise in noise_robustness.py uses 13 confusion pairs (l<->1, O<->0, ...). Here the
errors are counted instead: for each of the 60 example photos we align the ingredients section the
app extracted from the photo (after OCR + spelling correction, i.e. exactly what the NER model sees)
with the product's typed ingredient list, character by character, and count

    substitutions   a -> b   ("c" read as "e")
    deletions       a -> ""  (a letter dropped)
    insertions      a -> ab  (an extra letter after a)

for letters and digits. The result is saved as data/processed/learned_ocr_noise.json and used by
  - noisy_record()  : learned-noise copies of dataset records (evaluation set + training augmentation)
  - corrupt_token() : the same noise for one word

Only edits INSIDE words are simulated, so the number of tokens (and therefore the labels) stays the
same. Word merges and splits are counted and reported, but not simulated.

    python -m src.evaluation.learned_noise          # learn the model + print statistics
"""
import json
import random
import re
from collections import Counter, defaultdict

import pandas as pd

from src.evaluation.noise_robustness import _SUBSTITUTE
from src.preprocessing.pipeline import process_ingredient_text
from src.utils.config import project_path
from src.utils.io import read_jsonl

MODEL_PATH = project_path("data/processed/learned_ocr_noise.json")
MIN_SIMILARITY = 0.6     # photos whose section is too different from the typed list are skipped
MIN_COUNT = 20           # characters seen fewer times fall back to the global rates


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def align(ref: str, hyp: str) -> list:
    """Levenshtein alignment -> list of (ref_char or '', hyp_char or '') pairs."""
    n, m = len(ref), len(hyp)
    prev = list(range(m + 1))
    back = [[0] * (m + 1) for _ in range(n + 1)]     # 0 = match/sub, 1 = deletion, 2 = insertion
    for j in range(1, m + 1):
        back[0][j] = 2
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        back[i][0] = 1
        a = ref[i - 1]
        for j in range(1, m + 1):
            sub = prev[j - 1] + (a != hyp[j - 1])
            dele = prev[j] + 1
            ins = cur[j - 1] + 1
            best = min(sub, dele, ins)
            cur[j] = best
            back[i][j] = 0 if best == sub else (1 if best == dele else 2)
        prev = cur
    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        op = back[i][j]
        if op == 0:
            pairs.append((ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif op == 1:
            pairs.append((ref[i - 1], "")); i -= 1
        else:
            pairs.append(("", hyp[j - 1])); j -= 1
    return pairs[::-1]


def _trim(pairs: list) -> list:
    """Drop unmatched runs at both ends (text outside the section, or a list cut short)."""
    matched = [k for k, (a, b) in enumerate(pairs) if a and a == b]
    return pairs[matched[0]:matched[-1] + 1] if matched else []


def _long_runs(pairs: list, side: int, min_len: int = 3) -> set:
    """Positions inside runs of >= min_len consecutive deletions (side=1) or insertions (side=0):
    a dropped line or stray text, not a character error. Counted separately, not simulated."""
    skip, run = set(), []
    for k, pair in enumerate(pairs + [("x", "x")]):
        if pair[side] == "" and pair[1 - side] != "":
            run.append(k)
            continue
        if len(run) >= min_len:
            skip.update(run)
        run = []
    return skip


def learn(rows: list) -> dict:
    total = Counter()                        # how often each reference character was aligned
    subs = defaultdict(Counter)
    dels = Counter()
    ins = defaultdict(Counter)
    merges = splits = used = dropped_chars = stray_chars = 0
    per_photo = []
    for row in rows:
        ref, hyp = _clean(row["reference"]), _clean(row["section"])
        if not ref or not hyp:
            continue
        pairs = _trim(align(ref, hyp))
        if not pairs:
            continue
        errors = sum(a != b for a, b in pairs)
        similarity = 1 - errors / max(1, sum(1 for a, _ in pairs if a))
        per_photo.append({"product_id": row["product_id"], "similarity": round(similarity, 3)})
        if similarity < MIN_SIMILARITY:
            continue
        used += 1
        dropped = _long_runs(pairs, side=1)
        stray = _long_runs(pairs, side=0)
        dropped_chars += len(dropped)
        stray_chars += len(stray)
        last_ref = ""
        for k, (a, b) in enumerate(pairs):
            if k in dropped or k in stray:
                continue
            if a == " " and b == "":
                merges += 1
            elif a == "" and b == " ":
                splits += 1
            if a and a.isalnum():
                total[a] += 1
                if b == "":
                    dels[a] += 1
                elif b != a and b.isalnum():
                    subs[a][b] += 1
            if not a and b.isalnum() and last_ref.isalnum():
                ins[last_ref][b] += 1
            if a:
                last_ref = a
    n_total = sum(total.values())
    model = {
        "photos_used": used,
        "photos_aligned": len(per_photo),
        "characters": n_total,
        "global": {"sub": sum(sum(c.values()) for c in subs.values()) / n_total,
                   "del": sum(dels.values()) / n_total,
                   "ins": sum(sum(c.values()) for c in ins.values()) / n_total},
        "word_merges_per_1000_chars": round(1000 * merges / n_total, 2),
        "word_splits_per_1000_chars": round(1000 * splits / n_total, 2),
        "dropped_span_chars": dropped_chars,
        "stray_span_chars": stray_chars,
        "chars": {},
    }
    for ch, n in total.items():
        model["chars"][ch] = {"n": n, "sub": sum(subs[ch].values()) / n, "del": dels[ch] / n,
                              "ins": sum(ins[ch].values()) / n,
                              "sub_to": dict(subs[ch].most_common()), "ins_to": dict(ins[ch].most_common())}
    model["top_substitutions"] = [f"{a}->{b}: {c}" for (a, b), c in
                                  Counter({(a, b): c for a in subs for b, c in subs[a].items()}).most_common(15)]
    model["per_photo"] = per_photo
    return model


def load_model() -> dict:
    return json.loads(MODEL_PATH.read_text(encoding="utf-8"))


def _rates(model: dict, ch: str) -> dict:
    """Per-character rates; rare characters back off to the global rates and the hand-made pairs."""
    info = model["chars"].get(ch)
    if info and info["n"] >= MIN_COUNT and info["sub_to"]:
        return info
    g = model["global"]
    fallback = {b: 1 for b in _SUBSTITUTE.get(ch, [])}
    return {"sub": g["sub"] if fallback else 0.0, "del": g["del"], "ins": g["ins"],
            "sub_to": fallback, "ins_to": (info or {}).get("ins_to") or {"e": 1, "i": 1, "l": 1, "n": 1}}


def _pick(dist: dict, rng: random.Random) -> str:
    keys = list(dist)
    return rng.choices(keys, weights=[dist[k] for k in keys])[0]


def _same_case(new: str, old: str) -> str:
    return new.upper() if old.isupper() else new


def corrupt_token(token: str, model: dict, rng: random.Random, scale: float = 1.0) -> str:
    """Apply learned character errors to one word (never returns an empty word)."""
    if not token.isalnum():
        return token
    out = []
    for ch in token:
        r = _rates(model, ch.lower())
        u = rng.random()
        if u < scale * r["del"]:
            pass
        elif u < scale * (r["del"] + r["sub"]) and r["sub_to"]:
            out.append(_same_case(_pick(r["sub_to"], rng), ch))
        else:
            out.append(ch)
        if r["ins_to"] and rng.random() < scale * r["ins"]:
            out.append(_same_case(_pick(r["ins_to"], rng), ch))
    word = "".join(out)
    return word if word else token[0]


def noisy_record(record: dict, model: dict, rng: random.Random, scale: float = 1.0) -> dict:
    """Copy of a dataset record with learned noise; offsets and entity spans are recomputed.
    Percentage tokens are left alone. Returns the clean record if the noise would change the
    tokenisation (very rare), so labels always stay aligned."""
    text, offsets = record["text"], record["token_offsets"]
    pieces, new_offsets, pos, cursor = [], [], 0, 0
    for (s, e), tok in zip(offsets, record["tokens"]):
        pieces.append(text[cursor:s]); pos += s - cursor
        new_tok = tok if tok.endswith("%") else corrupt_token(tok, model, rng, scale)
        pieces.append(new_tok)
        new_offsets.append([pos, pos + len(new_tok)])
        pos += len(new_tok); cursor = e
    pieces.append(text[cursor:])
    new_text = "".join(pieces)
    check = process_ingredient_text(new_text)
    if check["text"] != new_text or len(check["tokens"]) != len(offsets):
        return record
    starts = {s: ns for (s, _), (ns, _) in zip(offsets, new_offsets)}
    ends = {e: ne for (_, e), (_, ne) in zip(offsets, new_offsets)}
    entities = []
    for ent in record["entities"]:
        ns, ne = starts[ent["start"]], ends[ent["end"]]
        entities.append({**ent, "start": ns, "end": ne, "text": new_text[ns:ne]})
    return {**record, "text": new_text, "tokens": check["tokens"], "token_offsets": new_offsets, "entities": entities}


def noisy_words(record: dict, model: dict, rng: random.Random, scale: float = 1.0) -> list:
    """Words only (for training): a learned-noise version of record['tokens']."""
    return [tok if tok.endswith("%") else corrupt_token(tok, model, rng, scale) for tok in record["tokens"]]


def main():
    ocr = {r["product_id"]: r for r in read_jsonl(project_path("data/images/ocr_results.jsonl"))}
    packets = pd.read_csv(project_path("data/images/packets.csv"), dtype={"product_id": str})
    rows = [{"product_id": p.product_id, "reference": p.reference_text, "section": ocr[p.product_id]["section"]}
            for p in packets.itertuples() if p.product_id in ocr]
    model = learn(rows)
    MODEL_PATH.write_text(json.dumps(model, indent=1, ensure_ascii=False), encoding="utf-8")
    g = model["global"]
    print(f"photos used {model['photos_used']} / {model['photos_aligned']} aligned; "
          f"{model['characters']:,} reference letters/digits")
    print(f"per character: substitution {g['sub']:.4f}, deletion {g['del']:.4f}, insertion {g['ins']:.4f} "
          f"(total {sum(g.values()):.4f})")
    print(f"word merges {model['word_merges_per_1000_chars']} / splits {model['word_splits_per_1000_chars']} "
          f"per 1000 characters (not simulated)")
    print(f"characters in dropped spans {model['dropped_span_chars']:,}, in stray spans {model['stray_span_chars']:,} "
          f"(missing/extra text, not simulated)")
    print("most common substitutions:", ", ".join(model["top_substitutions"]))
    print(f"saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
