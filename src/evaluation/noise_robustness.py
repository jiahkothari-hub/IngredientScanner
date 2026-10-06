"""How robust is the dictionary/rule baseline to OCR-like noise?

Usage:  python -m src.evaluation.noise_robustness

Person 3's pipeline feeds OCR text into the NER model, and OCR makes character mistakes. We take
the silver TEST texts, inject OCR-style character confusions at increasing rates, and compare the
baseline's predictions on the noisy text with its predictions on the clean text:

    F1(noisy predictions, clean predictions)   per noise rate and per label

The noise only SUBSTITUTES characters (same length), so entity spans stay comparable. A score of
1.0 means the noise changed nothing; the drop shows how brittle exact dictionary matching is - the
motivation for Person 2's Transformer models (and for their noise augmentation; `add_ocr_noise`
can be reused for that).
"""
import json
import random

from src.baseline.dictionary_ner import DictionaryNER
from src.evaluation.metrics import evaluate_entities
from src.preprocessing.normalize import normalize_text
from src.utils.config import project_path
from src.utils.io import read_jsonl

# Typical OCR confusions (each pair is applied in both directions)
OCR_CONFUSIONS = [("l", "1"), ("I", "l"), ("O", "0"), ("o", "0"), ("S", "5"), ("B", "8"), ("e", "c"),
                  ("i", "l"), ("a", "o"), ("n", "h"), ("u", "v"), ("g", "q"), ("Z", "2")]
# (punctuation confusions such as "," <-> "." are left out: they can create "..", which normalisation
#  would change, and then character offsets would no longer be comparable)
_SUBSTITUTE = {}
for a, b in OCR_CONFUSIONS:
    _SUBSTITUTE.setdefault(a, []).append(b)
    _SUBSTITUTE.setdefault(b, []).append(a)
NOISE_RATES = [0.0, 0.01, 0.02, 0.05, 0.10]
SAMPLE_SIZE = 1000


def add_ocr_noise(text: str, rate: float, rng: random.Random) -> str:
    """Replace each character that has a known OCR confusion with probability `rate`."""
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch in _SUBSTITUTE and rng.random() < rate:
            chars[i] = rng.choice(_SUBSTITUTE[ch])
    return "".join(chars)


def main():
    records = read_jsonl(project_path("data/splits/silver/test.jsonl.gz"))
    records = random.Random(0).sample(records, min(SAMPLE_SIZE, len(records)))
    ner = DictionaryNER()
    clean = [ner.predict_record(r) for r in records]
    results = {}
    for rate in NOISE_RATES:
        rng = random.Random(42)
        noisy_preds = []
        for r in records:
            noisy_text = add_ocr_noise(r["text"], rate, rng)
            assert normalize_text(noisy_text) == noisy_text and len(noisy_text) == len(r["text"])
            noisy_preds.append(ner.predict_record({**r, "text": noisy_text}))
        scores = evaluate_entities(clean, noisy_preds)
        results[str(rate)] = {"micro_f1": scores["micro"]["f1"],
                              "per_class_f1": {k: v["f1"] for k, v in scores["per_class"].items()}}
        print(f"noise rate {rate:.2f}: F1 vs clean predictions = {scores['micro']['f1']:.3f}")

    project_path("reports/noise_robustness.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # imported here (not at the top) so that the app, which reuses add_ocr_noise, needs no matplotlib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labels = ["INS_CODE", "SUGAR", "FAT", "ADDITIVE", "INGREDIENT"]
    colours = ["#1baf7a", "#eb6834", "#eda100", "#2a78d6", "#898781"]
    rates = [float(r) for r in results]
    for label, colour in zip(labels, colours):
        ys = [results[str(r)]["per_class_f1"][label] for r in rates]
        ax.plot([100 * r for r in rates], ys, marker="o", color=colour, linewidth=2, markersize=5, label=label)
    micro = [results[str(r)]["micro_f1"] for r in rates]
    ax.plot([100 * r for r in rates], micro, color="#0b0b0b", linewidth=2, linestyle="--", label="all entities")
    ax.set_xlabel("% of confusable characters replaced (OCR-style noise)")
    ax.set_ylabel("F1 vs. predictions on clean text")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.3, 100 * rates[-1] + 0.3)
    ax.set_title("Rule baseline under OCR-like noise", loc="left", color="#0b0b0b")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(True, color="#e6e5e0", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(project_path("reports/figures/11_noise_robustness.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
