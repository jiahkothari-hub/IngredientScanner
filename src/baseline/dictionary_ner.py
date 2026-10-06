"""Dictionary / rule-based NER baseline.

Input: an ingredient list (any text).  Output: entities (character spans + labels) or BIO tags.

The baseline uses exactly the components built for weak labelling: normalisation, dictionaries
(curated + Open Food Facts taxonomy), INS/E-number regular expressions and head-word rules.
So in this project "baseline" and "silver labeller" are the same system, which is normal in weak
supervision: the question for Person 2 is whether a Transformer trained on these noisy labels
GENERALISES beyond the rules (unseen spellings, OCR noise, missing commas). Both are evaluated on
the human-verified gold test set, which the rules have never been tuned on.

    from src.baseline.dictionary_ner import DictionaryNER
    ner = DictionaryNER()
    ner.predict("Sugar, Acidity regulator (INS 330)")["entities"]
"""
from src.labeling.bio import entities_to_bio
from src.labeling.weak_labeler import label_processed, label_text
from src.preprocessing.pipeline import process_ingredient_text


class DictionaryNER:
    name = "dictionary_rules_v1"

    def predict(self, text: str) -> dict:
        """Raw text -> {'text': normalised text, 'tokens', 'token_offsets', 'entities'}."""
        return label_text(text)

    def predict_record(self, record: dict) -> list:
        """Predict entities for a dataset record (uses its already-normalised text, so offsets match)."""
        processed = process_ingredient_text(record["text"])
        if processed["text"] != record["text"]:
            raise ValueError(f"record {record.get('id')} text is not normalised")
        return [{k: e[k] for k in ("start", "end", "text", "label")} for e in label_processed(processed)]

    def predict_tags(self, record: dict) -> list:
        return entities_to_bio(record["token_offsets"], self.predict_record(record))
