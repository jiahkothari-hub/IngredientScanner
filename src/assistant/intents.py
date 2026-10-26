"""Intent classification for the ingredient assistant: what kind of question is this?

    from src.assistant.intents import classify
    classify("why is citric acid added?")   -> ("why_added", 0.71)

Two classifiers, compared in src/assistant/evaluate.py on held-out test questions:
  rules  keyword patterns (the baseline)
  model  TF-IDF of words (1-2 grams) and character 2-5 grams + logistic regression, trained on
         data/assistant/intent_examples.csv (about 15 hand-written examples per intent). Character
         n-grams make it tolerant to typos ("vegetarain", "alergens").

Medical questions are the one intent where a miss matters: the assistant must not answer "can I eat
this with diabetes?" as if it were a portion question. So the keyword safety net below overrides the
model whenever a health condition or a health judgement is named.
"""
import re
from functools import lru_cache

import pandas as pd

from src.utils.config import project_path

EXAMPLES = project_path("data/assistant/intent_examples.csv")
INTENTS = ["what_is", "why_added", "diet", "contains", "allergens", "list_category", "portion", "daily_limit",
           "regulatory", "medical", "smalltalk"]

MEDICAL_RE = re.compile(
    r"diabet|pregnan|breastfeed|cancer|carcinogen|kidney|liver|heart|blood\s*pressure|hypertension|cholesterol|"
    r"\bpcos\b|thyroid|medicat|medicine|doctor|disease|illness|sick|\bill\b|hyperactiv|\badhd\b|autis|"
    r"\bhealthy\b|unhealthy|good\s+for\s+(?:me|you|my|health|weight|kids|children)|bad\s+for|weight\s+loss|"
    r"lose\s+weight|make\s+me\s+fat|\bbaby\b|infant|toddler|safe\s+for\s+(?:my|kids|children|a\s+child)|"
    r"should\s+i\s+(?:stop|eat|avoid)|\bdiet\s+plan\b", re.IGNORECASE)

RULES = [  # (intent, pattern), first match wins
    ("smalltalk", r"^\s*(hi|hello|hey|thanks?|thank\s+you|bye|ok|okay|good\s+(morning|evening)|help)\b|"
                  r"what\s+can\s+you\s+do|who\s+are\s+you|how\s+do\s+i\s+use|what\s+questions"),
    ("regulatory", r"\bbann?ed\b|\bban\b|illegal|\blegal\b|(?:allowed|permitted|approved|sold)\s+(?:in|by)\b|"
                   r"\bfda\b|fssai|europe|\beu\b|warning\s+label|countries"),
    ("daily_limit", r"\badi\b|acceptable\s+daily|daily\s+(limit|intake)\s+(of|for)|how\s+much\s+\w+(\s+\w+)?\s+"
                    r"(is\s+)?(safe|allowed|can\s+i\s+have)|per\s+kg|safe\s+amount|maximum\s+daily|mg\s+.*per\s+day"),
    ("portion", r"how\s+much\s+(sugar|salt|fat|saturated|energy)|calories|teaspoons|portion|serving|"
                r"daily\s+(sugar|limit)|too\s+much|high\s+in|whole\s+pack|amount\s+ok|percentage"),
    ("allergens", r"allerg|gluten|dairy|lactose|nuts?\b|peanut|sesame|\beggs?\b|soy\b|may\s+contain|traces?"),
    ("diet", r"vegan|vegetarian|halal|kosher|\bjain|animal|plant\s+based"),
    ("list_category", r"which\s+(ingredients|ones|colours|additives)|\blist\b|show\s+me|hidden\s+sugars?|"
                      r"how\s+many|what\s+(preservatives|sweeteners|fats|oils|flavourings|colours|sugars|additives)|"
                      r"all\s+the|e\s*numbers"),
    ("contains", r"does\s+(this|it|the\s+product)\s+(contain|have|use)|is\s+there|any\s+\w+\s+in\s+(this|it)|"
                 r"one\s+of\s+the\s+ingredients|have\s+they\s+used"),
    ("why_added", r"\bwhy\b|used\s+for|purpose|function|what\s+does\s+.+\s+do\b|role|job"),
    ("what_is", r"what\s+(is|are|does|'s)|whats|tell\s+me\s+about|explain|meaning|define"),
]


def rule_intent(question: str) -> str:
    """Keyword baseline."""
    q = question.lower()
    if MEDICAL_RE.search(q):
        return "medical"
    for intent, pattern in RULES:
        if re.search(pattern, q):
            return intent
    return "what_is"


@lru_cache(maxsize=1)
def model():
    """Train the intent model (well under a second, so it is trained on first use instead of stored)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline
    data = pd.read_csv(EXAMPLES)
    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
    ])
    pipe = Pipeline([("features", features), ("clf", LogisticRegression(C=10, max_iter=2000))])
    return pipe.fit(data["question"].str.lower(), data["intent"])


def model_intent(question: str) -> tuple:
    """(intent, probability) from the trained model alone."""
    m = model()
    probs = m.predict_proba([question.lower()])[0]
    best = probs.argmax()
    return m.classes_[best], float(probs[best])


def classify(question: str) -> tuple:
    """(intent, confidence): the model, with the medical keyword safety net on top."""
    if MEDICAL_RE.search(question):
        return "medical", 1.0
    return model_intent(question)
