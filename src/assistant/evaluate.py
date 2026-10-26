"""Evaluate the ingredient assistant on held-out test questions.

    python -m src.assistant.evaluate        -> reports/assistant_evaluation.md / .json

Two question sets, both written separately from the intent training examples, with typos, paraphrases,
follow-up questions that need "it" resolved, and health questions that must be declined. For each question
we know the intent, the ingredient it is about, phrases the answer must contain (taken from the product's
ingredient list and the knowledge base), and whether it must be declined.
  development  data/assistant/test_questions.jsonl, 64 questions about 10 products. Its first run found
               four general bugs, which were then fixed, so its final score is optimistic. The first-run
               score is kept in FIRST_RUN below.
  fresh        data/assistant/test_questions_fresh.jsonl, 24 questions (partly about 5 other products),
               written after those fixes and evaluated without any change afterwards: the honest estimate.

Measured:
  intent accuracy        keyword rules (baseline) vs. the trained model vs. model + medical safety net
  mention accuracy       did the assistant pick the right ingredient? (questions that name or refer to one)
  answer correctness     does the answer contain every required phrase?
  refusals               precision / recall of declining medical questions
  follow-ups             answer correctness on the questions that need "it" resolved from history
Caveat: the questions were written by the developers, so they are not a random sample of real users.
"""
import json
import re
from collections import Counter, defaultdict

from src.assistant.assistant import IngredientAssistant
from src.assistant.intents import model_intent, rule_intent
from src.utils.config import project_path
from src.utils.io import read_jsonl

QUESTIONS = project_path("data/assistant/test_questions.jsonl")
FRESH_QUESTIONS = project_path("data/assistant/test_questions_fresh.jsonl")
# development set, first run, before the bugs it revealed were fixed
FIRST_RUN = {"answer_correct": 0.875, "intent_model_plus_safety_net": 0.906, "intent_rules": 0.672,
             "mention_accuracy": 0.969, "refusal_recall": 1.0, "refusal_precision": 1.0, "followups": 0.8}
# fresh set, its one untouched run. One change came after it (an allergen question must name an allergen;
# found by a unit test), so the current fresh score is reported next to this one.
FRESH_FIRST_RUN = {"answer_correct": 0.917, "intent_model_plus_safety_net": 0.833, "intent_rules": 0.833,
                   "mention_accuracy": 1.0, "refusal_recall": 1.0, "refusal_precision": 1.0, "followups": 0.5}
PRODUCTS = project_path("data/test_samples/typed_products.json")
REPORT = project_path("reports/assistant_evaluation")
DRINK_RE = re.compile(r"carbonated|water,|energy drink|\bcola\b", re.IGNORECASE)


def plain(text: str) -> str:
    return re.sub(r"\*\*|[“”]", "", text).lower()


def product_contexts(names: set) -> dict:
    """Scan each sample product's typed list and build the daily-intake guide the app would show."""
    from src.app.intake import PROFILES, assess_portion
    from src.app.nutrition import estimate_from_similar_product
    from src.app.scanner import IngredientScanner
    texts = {p["product"]: p["ingredients_text"] for p in json.loads(PRODUCTS.read_text(encoding="utf-8"))["products"]}
    scanner = IngredientScanner()
    out = {}
    for name in names:
        result = scanner.scan_text(texts[name])
        drink = bool(DRINK_RE.search(texts[name]))
        portion = 250.0 if drink else 30.0
        per100 = (result.get("nutrition") or {}).get("per_100g") or estimate_from_similar_product(result["input_text"]).get("per_100g", {})
        rows = assess_portion(per100, portion, PROFILES["Adult (2000 kcal)"], "drink" if drink else "food") if per100 else []
        out[name] = (result, {"rows": rows, "portion": portion, "drink": drink})
    return out


def evaluate(path=QUESTIONS) -> dict:
    questions = read_jsonl(path)
    contexts = product_contexts({q["product"] for q in questions if q["product"]})
    rows = []
    for q in questions:
        result, guide = contexts.get(q["product"], (None, None))
        bot = IngredientAssistant(result, guide, body_weight_kg=60)
        history = []
        for earlier in q["history"]:
            history.append(bot.answer(earlier, history))
        reply = bot.answer(q["question"], history)
        text = plain(reply["text"])
        missing = [p for p in q["must_include"] if p.lower() not in text]
        mention_ok = None
        if q["mention"]:
            mention_ok = any(q["mention"].lower() in m.lower() for m in reply["mentions"]) or \
                         q["mention"].lower() in text
        rows.append({"id": q["id"], "product": q["product"], "question": q["question"], "history": q["history"],
                     "intent": q["intent"], "intent_rules": rule_intent(q["question"]),
                     "intent_model": model_intent(q["question"])[0], "intent_final": reply["intent"],
                     "mention": q["mention"], "mention_found": reply["mentions"], "mention_ok": mention_ok,
                     "answer_ok": not missing, "missing": missing, "refuse": q["refuse"], "refused": reply["refused"],
                     "answer": reply["text"]})
    return summarise(rows)


def summarise(rows: list) -> dict:
    n = len(rows)
    acc = lambda key: round(sum(r[key] == r["intent"] for r in rows) / n, 3)
    with_mention = [r for r in rows if r["mention_ok"] is not None]
    followups = [r for r in rows if r["history"]]
    should, did = [r for r in rows if r["refuse"]], [r for r in rows if r["refused"]]
    per_intent = defaultdict(lambda: [0, 0])
    for r in rows:
        per_intent[r["intent"]][0] += r["answer_ok"]
        per_intent[r["intent"]][1] += 1
    return {
        "questions": n,
        "intent_accuracy": {"rules": acc("intent_rules"), "model": acc("intent_model"),
                            "model_plus_safety_net": acc("intent_final")},
        "mention_accuracy": round(sum(r["mention_ok"] for r in with_mention) / len(with_mention), 3),
        "mention_questions": len(with_mention),
        "answer_correct": round(sum(r["answer_ok"] for r in rows) / n, 3),
        "refusal": {"precision": round(sum(r["refuse"] for r in did) / max(1, len(did)), 3),
                    "recall": round(sum(r["refused"] for r in should) / max(1, len(should)), 3),
                    "medical_questions": len(should)},
        "followups": {"questions": len(followups),
                      "answer_correct": round(sum(r["answer_ok"] for r in followups) / max(1, len(followups)), 3)},
        "per_intent_answer_correct": {k: f"{v[0]}/{v[1]}" for k, v in sorted(per_intent.items())},
        "intent_confusions_model": dict(Counter(f"{r['intent']} -> {r['intent_final']}" for r in rows
                                                 if r["intent_final"] != r["intent"])),
        "rows": rows,
    }


def write_report(dev: dict, fresh: dict):
    f, ff = FIRST_RUN, FRESH_FIRST_RUN
    pct = lambda x: f"{x:.1%}"
    lines = ["# Ingredient assistant evaluation (auto-generated by `python -m src.assistant.evaluate`)", "",
             "Questions written by the developers, separately from the intent training examples (not a random "
             "sample of real users). The development set's first run revealed four general bugs that were then "
             "fixed, so its final column is optimistic. The **fresh set** was written after those fixes and its "
             "first run is the honest estimate. One later change (an allergen question must name an allergen, "
             "found by a unit test) is included in the 'current' column.", "",
             "| Measure | Dev, first run | Dev, final | **Fresh, first run** | Fresh, current |",
             "|---|---:|---:|---:|---:|",
             f"| Questions | {dev['questions']} | {dev['questions']} | {fresh['questions']} | {fresh['questions']} |"]
    rows = [("Intent accuracy: keyword rules (baseline)", f["intent_rules"], dev["intent_accuracy"]["rules"],
             ff["intent_rules"], fresh["intent_accuracy"]["rules"]),
            ("Intent accuracy: TF-IDF + logistic regression + safety net", f["intent_model_plus_safety_net"],
             dev["intent_accuracy"]["model_plus_safety_net"], ff["intent_model_plus_safety_net"],
             fresh["intent_accuracy"]["model_plus_safety_net"]),
            ("Right ingredient found", f["mention_accuracy"], dev["mention_accuracy"], ff["mention_accuracy"],
             fresh["mention_accuracy"]),
            ("Answer contains every required fact", f["answer_correct"], dev["answer_correct"], ff["answer_correct"],
             fresh["answer_correct"]),
            ("Medical questions declined (recall)", f["refusal_recall"], dev["refusal"]["recall"], ff["refusal_recall"],
             fresh["refusal"]["recall"]),
            ("Declined answers that were medical (precision)", f["refusal_precision"], dev["refusal"]["precision"],
             ff["refusal_precision"], fresh["refusal"]["precision"]),
            ("Follow-up questions (\"it\" from history)", f["followups"], dev["followups"]["answer_correct"],
             ff["followups"], fresh["followups"]["answer_correct"])]
    lines += [f"| {name} | {pct(a)} | {pct(b)} | **{pct(c)}** | {pct(d)} |" for name, a, b, c, d in rows]
    lines += ["", f"Fresh set: {fresh['refusal']['medical_questions']} medical and {fresh['followups']['questions']} "
              f"follow-up questions only, so those two rows are small samples.", ""]
    for title, s in (("Fresh set", fresh), ("Development set (final)", dev)):
        failures = [r for r in s["rows"] if not r["answer_ok"]]
        lines += [f"## {title}: failures ({len(failures)})", "",
                  "Answer correctness per intent: " + ", ".join(f"{k} {v}" for k, v in s["per_intent_answer_correct"].items()), ""]
        for r in failures:
            lines.append(f"* **#{r['id']}** ({r['product'] or 'no product'}) “{r['question']}” - intent "
                         f"{r['intent_final']} (expected {r['intent']}); missing {r['missing']}. "
                         f"Answer: {plain(r['answer'])[:240]}")
        lines.append("")
    lines += ["Notes: #102 is a scoring artefact (the answer says “anti caking agent”, the expected phrase was "
              "“anti-caking”); it is counted as a failure because the fresh set was not changed after its run. "
              "#14 and #102's carrier wording come from the knowledge base itself: Open Food Facts lists citric acid "
              "as antioxidant/sequestrant (not acidity regulator) and silicon dioxide as a carrier."]
    (REPORT.parent / f"{REPORT.name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REPORT.parent / f"{REPORT.name}.json").write_text(
        json.dumps({"first_run_development": FIRST_RUN, "development": dev, "fresh": fresh}, indent=2,
                   ensure_ascii=False), encoding="utf-8")


def main():
    dev, fresh = evaluate(QUESTIONS), evaluate(FRESH_QUESTIONS)
    write_report(dev, fresh)
    for name, s in (("development", dev), ("fresh", fresh)):
        ia = s["intent_accuracy"]
        print(f"{name:11}: intent rules {ia['rules']:.3f} / model {ia['model_plus_safety_net']:.3f}; mention "
              f"{s['mention_accuracy']:.3f}; answers {s['answer_correct']:.3f}; refusal recall {s['refusal']['recall']:.3f} "
              f"precision {s['refusal']['precision']:.3f}; follow-ups {s['followups']['answer_correct']:.3f}")
    print(f"saved {REPORT}.md")


if __name__ == "__main__":
    main()
