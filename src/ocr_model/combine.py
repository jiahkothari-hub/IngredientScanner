"""Combine two OCR readers with food-language knowledge: RapidOCR + our fine-tuned CRNN.

    python -m src.ocr_model.combine                (after src.ocr_model.finetune)

The two recognisers make different mistakes (RapidOCR: case, merged words, odd symbols; our CRNN: a/o, blurred
print). When their readings of a line differ, we keep the one that looks more like an ingredient list:
  plausibility = average log-probability per character under a character 6-gram language model
                 + share of words that are known words
Both the language model and the word list come from the TRAIN split of our ingredient lists only (+ general
English words); the 60 photo products are all in the silver TEST split, so none of their text is used.

Two ways of combining:
  line : take one whole reading
  word : when both readings have the same number of words, choose word by word; otherwise as "line"
RapidOCR is the default reader; the CRNN's version replaces it only when it is more plausible by more than a
margin. The margin is chosen on one photo half and applied to the other (same two-fold split as finetune.py),
so the test photos never influence their own setting. "Never replace" is always a candidate, so the combination
cannot be chosen to be worse than RapidOCR on the half where it is chosen.
"""
import gzip
import json
import math
import re
from collections import Counter
from functools import lru_cache

import numpy as np

from src.ocr_model.charset import clean_text
from src.ocr_model.evaluate import error_rates
from src.utils.config import project_path

ORDER = 6
WEIGHTS = (0.35, 0.25, 0.18, 0.12, 0.07, 0.03)     # longest context first; fixed in advance, not tuned on photos
MARGINS = (0.0, 0.1, 0.25, 0.5, 1.0, math.inf)     # inf = never replace RapidOCR
WORD_RE = re.compile(r"[A-Za-z]{3,}")


class CharLM:
    """Character n-gram model with fixed-weight interpolation of orders 6..1 (+ add-one at order 1)."""

    def __init__(self, texts):
        self.counts = [Counter() for _ in range(ORDER + 1)]
        for t in texts:
            t = "\x02" * (ORDER - 1) + t.lower() + "\x03"
            for n in range(1, ORDER + 1):
                for i in range(ORDER - 1, len(t)):
                    if i - n + 1 >= 0:
                        self.counts[n][t[i - n + 1:i + 1]] += 1
        self.vocab = len({k for k in self.counts[1]}) + 1
        self.total = sum(self.counts[1].values())

    def prob(self, history: str, ch: str) -> float:
        p = 0.0
        for w, n in zip(WEIGHTS, range(ORDER, 0, -1)):
            ctx = history[len(history) - (n - 1):] if n > 1 else ""
            if n == 1:
                p += w * (self.counts[1][ch] + 1) / (self.total + self.vocab)
                continue
            denom = self.counts[n - 1][ctx]
            if denom:
                p += w * self.counts[n][ctx + ch] / denom
        return p

    def score(self, text: str) -> float:
        """Average natural-log probability per character (higher = more ingredient-like)."""
        t = "\x02" * (ORDER - 1) + text.lower() + "\x03"
        logs = [math.log(max(self.prob(t[:i], t[i]), 1e-12)) for i in range(ORDER - 1, len(t))]
        return sum(logs) / len(logs)


@lru_cache(maxsize=1)
def resources():
    with gzip.open(project_path("data/splits/silver/train.jsonl.gz"), "rt", encoding="utf-8") as f:
        texts = [clean_text(json.loads(line)["text"]) for line in f]
    lm = CharLM(texts)
    words = Counter(w.lower() for t in texts for w in WORD_RE.findall(t))
    known = {w for w, c in words.items() if c >= 2}
    try:
        import symspellpy
        from pathlib import Path
        english = Path(symspellpy.__file__).parent / "frequency_dictionary_en_82_765.txt"
        known |= {line.split(" ", 1)[0] for line in english.read_text(encoding="utf-8").splitlines()}
    except ImportError:
        pass
    return lm, frozenset(known)


def plausibility(text: str) -> float:
    lm, known = resources()
    if not text.strip():
        return -99.0
    words = WORD_RE.findall(text)
    share = sum(w.lower() in known for w in words) / len(words) if words else 0.5
    return lm.score(text) + share


def combine_line(rapid: str, ours: str, margin: float) -> str:
    if rapid == ours or margin == math.inf:
        return rapid
    return ours if plausibility(ours) > plausibility(rapid) + margin else rapid


def combine_words(rapid: str, ours: str, margin: float) -> str:
    a, b = rapid.split(), ours.split()
    if rapid == ours or margin == math.inf:
        return rapid
    if len(a) != len(b):
        return combine_line(rapid, ours, margin)
    out = []
    for i, (x, y) in enumerate(zip(a, b)):
        if x == y:
            out.append(x)
            continue
        context = " ".join(out[-3:])                      # judge the word in its left context
        sx = plausibility((context + " " + x).strip())
        sy = plausibility((context + " " + y).strip())
        out.append(y if sy > sx + margin else x)
    return " ".join(out)


def run(crnn: str = "finetuned") -> dict:
    """crnn: "finetuned" (CRNN v2 + real-crop fine-tuning, greedy) or "improved" (+ Open Food Facts crops + beam
    search with the food LM, see improve_crnn.py). Either way every CRNN reading comes from a fold model that
    never saw the photo."""
    from rapidfuzz.distance import Levenshtein
    from src.ocr_model.finetune import photo_halves
    if crnn == "improved":
        from src.ocr_model.improve_crnn import improved_readings
        rows = improved_readings().rename(columns={"crnn_improved": "crnn_ft"})
    else:
        from src.ocr_model.compare_real import crop_readings
        rows = crop_readings()
    suffix = "" if crnn == "finetuned" else f"_{crnn}"
    halves = photo_halves(rows, 0)
    labels = rows["label"].tolist()
    methods = {"line": combine_line, "word": combine_words}
    result = {"crops": len(rows), "photos": rows["product_id"].nunique(), "methods": {}}
    for name, fn in methods.items():
        final = [None] * len(rows)
        chosen = []
        for choose_on, apply_to in [(halves[0], halves[1]), (halves[1], halves[0])]:
            pick = rows["product_id"].isin(choose_on).to_numpy()
            use = rows["product_id"].isin(apply_to).to_numpy()
            scores = {}
            for m in MARGINS:
                preds = [fn(r, o, m) for r, o in zip(rows.loc[pick, "rapidocr"], rows.loc[pick, "crnn_ft"])]
                scores[m] = error_rates(preds, rows.loc[pick, "label"].tolist())["cer"]
            best = min(MARGINS, key=lambda m: (scores[m], -m if m != math.inf else -1e9))
            chosen.append({"margin": best, "cer_on_choice_half": scores})
            for i in np.flatnonzero(use):
                final[i] = fn(rows.at[i, "rapidocr"], rows.at[i, "crnn_ft"], best)
        exact = error_rates(final, labels)
        nocase = error_rates([p.lower() for p in final], [l.lower() for l in labels])
        replaced = sum(f != r for f, r in zip(final, rows["rapidocr"]))
        result["methods"][name] = {"cer": exact["cer"], "wer": exact["wer"], "line_accuracy": exact["line_accuracy"],
                                   "cer_ignoring_case": nocase["cer"], "lines_changed_from_rapidocr": replaced,
                                   "chosen_margins": [{"margin": str(c["margin"]),
                                                       "cer_on_choice_half": {str(k): v for k, v in c["cer_on_choice_half"].items()}}
                                                      for c in chosen]}
        rows[f"combined_{name}"] = final
    for s in ("rapidocr", "crnn_ft"):
        exact = error_rates(rows[s].tolist(), labels)
        nocase = error_rates(rows[s].str.lower().tolist(), [l.lower() for l in labels])
        result[s] = {"cer": exact["cer"], "wer": exact["wer"], "line_accuracy": exact["line_accuracy"],
                     "cer_ignoring_case": nocase["cer"]}
    best = min(result["methods"], key=lambda k: result["methods"][k]["cer"])
    col = f"combined_{best}"
    better = rows[[a != b for a, b in zip(rows[col], rows["rapidocr"])]]
    result["changed_examples"] = [{"label": r.label, "rapidocr": r.rapidocr, "crnn": r.crnn_ft, "combined": getattr(r, col)}
                                  for r in better.head(30).itertuples()]
    result["best_method"] = best
    result["crnn_version"] = crnn
    oracle = [a if Levenshtein.distance(a, l) <= Levenshtein.distance(b, l) else b
              for a, b, l in zip(rows["rapidocr"], rows["crnn_ft"], labels)]
    result["oracle_line_cer"] = error_rates(oracle, labels)["cer"]
    result["lines_crnn_strictly_better"] = sum(Levenshtein.distance(b, l) < Levenshtein.distance(a, l)
                                               for a, b, l in zip(rows["rapidocr"], rows["crnn_ft"], labels))
    project_path(f"reports/ocr_model/combined{suffix}.json").write_text(json.dumps(result, indent=2))
    report(result, suffix)
    return result


def report(result: dict, suffix: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    best = result["methods"][result["best_method"]]
    lines = ["# RapidOCR + our CRNN, combined with a food language model", "",
             f"{result['crops']} real crops from {result['photos']} photos. Margins chosen on the other photo half.", "",
             "| Reader | CER | CER (ignoring case) | WER | Lines exact |", "|---|---|---|---|---|"]
    rows = [("RapidOCR alone", result["rapidocr"]), ("Our CRNN alone (fine-tuned)", result["crnn_ft"])]
    rows += [(f"Combined ({k}-level)", v) for k, v in result["methods"].items()]
    for name, m in rows:
        lines.append(f"| {name} | {m['cer']:.3f} | {m['cer_ignoring_case']:.3f} | {m['wer']:.3f} | {m['line_accuracy']:.1%} |")
    lines += ["", "Chosen margins: " + "; ".join(f"{k}: " + ", ".join(c["margin"] for c in v["chosen_margins"])
                                                for k, v in result["methods"].items()),
              f"Lines changed from RapidOCR's reading ({result['best_method']}-level): {best['lines_changed_from_rapidocr']}",
              "", "## Lines where the combination differs from RapidOCR", "",
              "| Label | RapidOCR | CRNN | Combined |", "|---|---|---|---|"]
    lines += [f"| `{e['label']}` | `{e['rapidocr']}` | `{e['crnn']}` | `{e['combined']}` |" for e in result["changed_examples"]]
    lines.insert(4, f"CRNN version: {result['crnn_version']}. Perfect-chooser ceiling: CER {result['oracle_line_cer']:.4f}; "
                    f"CRNN strictly better on {result['lines_crnn_strictly_better']} lines.\n")
    project_path(f"reports/ocr_model/combined{suffix}.md").write_text("\n".join(lines), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 3.4))
    names = [n for n, _ in rows]
    vals = [m["cer"] for _, m in rows]
    colours = ["#9FB9B2", "#C9B8A6"] + ["#0E7C66"] * (len(rows) - 2)
    bars = ax.barh(names[::-1], vals[::-1], color=colours[::-1])
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xlabel("character error rate on real crops (lower is better)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(project_path(f"reports/figures/ocr_model/combined_cer{suffix}.png"), dpi=160)
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--crnn", default="finetuned", choices=["finetuned", "improved"])
    r = run(parser.parse_args().crnn)
    print(f"perfect-chooser ceiling {r['oracle_line_cer']:.4f}; CRNN strictly better on "
          f"{r['lines_crnn_strictly_better']} of {r['crops']} lines")
    for name, m in [("RapidOCR", r["rapidocr"]), ("CRNN", r["crnn_ft"])] + [(f"combined-{k}", v) for k, v in r["methods"].items()]:
        print(f"{name:16s} CER {m['cer']:.4f}  CER(no case) {m['cer_ignoring_case']:.4f}  exact {m['line_accuracy']:.1%}")
    for k, v in r["methods"].items():
        print(k, "margins chosen:", [c["margin"] for c in v["chosen_margins"]], "lines changed:", v["lines_changed_from_rapidocr"])


if __name__ == "__main__":
    main()
