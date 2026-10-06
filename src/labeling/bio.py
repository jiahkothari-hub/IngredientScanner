"""Convert between character-level entity spans and word-level BIO tags.

BIO scheme: B-X = first token of an entity of type X, I-X = following tokens, O = outside.
    tokens:  Acidity      regulator   (  INS        330        )
    tags:    B-FUNCTION_CLASS I-FUNCTION_CLASS O  B-INS_CODE I-INS_CODE O

Entities are stored as character spans because they are independent of any tokenizer;
BIO tags are what token-classification models (Person 2) are trained on.
"""

LABELS = ["SUGAR", "SWEETENER", "FAT", "PRESERVATIVE", "COLOUR", "ADDITIVE", "INS_CODE",
          "FUNCTION_CLASS", "FLAVOURING", "INGREDIENT"]
BIO_TAGS = ["O"] + [f"{prefix}-{label}" for label in LABELS for prefix in ("B", "I")]
LABEL2ID = {tag: i for i, tag in enumerate(BIO_TAGS)}
ID2LABEL = {i: tag for tag, i in LABEL2ID.items()}


def entities_to_bio(token_offsets: list, entities: list) -> list:
    """Assign one BIO tag per token. A token belongs to an entity if it lies inside the entity span.

    Raises ValueError if an entity boundary cuts through a token (that would be a bug in the
    labeller or annotation, and silently fixing it would hide the problem).
    """
    tags = ["O"] * len(token_offsets)
    for ent in sorted(entities, key=lambda e: e["start"]):
        inside = [i for i, (s, e) in enumerate(token_offsets) if s >= ent["start"] and e <= ent["end"]]
        cut = [i for i, (s, e) in enumerate(token_offsets)
               if s < ent["end"] and e > ent["start"] and i not in inside]
        if cut or not inside:
            raise ValueError(f"Entity {ent.get('text')!r} [{ent['start']}:{ent['end']}] is not aligned to tokens")
        for n, i in enumerate(inside):
            if tags[i] != "O":
                raise ValueError(f"Overlapping entities at token {i} ({ent.get('text')!r})")
            tags[i] = ("B-" if n == 0 else "I-") + ent["label"]
    return tags


def bio_to_entities(token_offsets: list, tags: list, text: str = None) -> list:
    """Inverse of entities_to_bio. An I- tag that does not continue an entity of the same type is
    treated as the start of a new entity (the usual lenient decoding, as in seqeval)."""
    entities, current = [], None
    for (start, end), tag in zip(token_offsets, tags):
        prefix, _, label = tag.partition("-")
        if prefix == "B" or (prefix == "I" and (current is None or current["label"] != label)):
            if current:
                entities.append(current)
            current = {"start": start, "end": end, "label": label}
        elif prefix == "I":
            current["end"] = end
        else:
            if current:
                entities.append(current)
            current = None
    if current:
        entities.append(current)
    if text is not None:
        for ent in entities:
            ent["text"] = text[ent["start"]:ent["end"]]
    return entities
