"""Weak labeller: dictionaries + regex rules -> entity spans (silver labels).

    from src.labeling.weak_labeler import label_text
    label_text("Sugar, glucose syrup, Acidity regulator (INS 330), palm oil")["entities"]
    -> SUGAR "Sugar", SUGAR "glucose syrup", FUNCTION_CLASS "Acidity regulator",
       INS_CODE "INS 330", FAT "palm oil"

Decision order (the first step that applies wins, see reports/entity_schema.md):
  0. Skip allergen statements / storage advice ("Contains: milk", "May contain nuts").
  1. Additive codes (regex)                                     -> INS_CODE
  2. Split the list into segments (commas, brackets, ...); inside each segment cut out the codes
     and remove framing words ("contains less than 2% of", trailing "20%").
  3. The whole piece is a dictionary term                       -> that term's label
  4. The piece is several dictionary terms joined by and/&/or   -> one entity each
  5. Two or more dictionary terms covering the piece ("Emulsifier soy lecithin") -> one entity each
  6. The piece ENDS with a dictionary term ("organic cane sugar") or a head word
     ("refined palm oil", "natural strawberry flavour")          -> label of the head
  7. Otherwise a short Latin-script piece whose words all occur in >= 3 products
     (filters OCR garbage; not required if the whole list is one segment) -> INGREDIENT (fallback)
"""
import re
from functools import lru_cache

from src.labeling.lexicon import load_lexicon, lookup_key
from src.labeling.rules import (ADDITIVE_LABELS, ADDITIVE_MODIFIERS, CONNECTOR_RE, SUFFIX_MAX_PREFIX_WORDS, FRAMING_PREFIX_RE, SKIP_PIECE_RE, STATEMENT_START_RE, STOPWORDS,
                                TRAILING_NOISE_RE, head_rule)
from src.preprocessing.normalize import non_latin_letter_ratio
from src.preprocessing.pipeline import process_ingredient_text
from src.utils.config import project_path

MAX_FALLBACK_WORDS = 6
WORD_RE = re.compile(r"[^\W_]+(?:['-][^\W_]+)*")


VOCAB_PATH = project_path("data/processed/ingredient_vocabulary.txt")


@lru_cache(maxsize=1)
def corpus_vocabulary():
    """Words seen in >= 3 products (src/labeling/build_vocabulary.py). Empty set if not built yet."""
    if not VOCAB_PATH.exists():
        return frozenset()
    lines = VOCAB_PATH.read_text(encoding="utf-8").splitlines()
    return frozenset(line.split("\t")[0] for line in lines if line and not line.startswith("#"))


def _known_words(piece: str) -> bool:
    """True if every word of 3+ letters is in the corpus vocabulary (or no vocabulary is available)."""
    vocabulary = corpus_vocabulary()
    if not vocabulary:
        return True
    return all(w.lower() in vocabulary for w in re.findall(r"[^\W\d_]{3,}", piece))


SEPARATOR_RE = re.compile(r"[,;:(\[]|\.(?=\s)")   # a full stop only when followed by a space ("0.08%" is not)
# a statement can also start right after a closing bracket ("... (citric acid) CONTAINS PHENYLALANINE")
STATEMENT_BOUNDARY_RE = re.compile(r"[,;:(\[)\]]|\.(?=\s)")


def find_statement_spans(text: str) -> list:
    """Character spans of allergen/storage/contact statements.

    A span starts at the previous separator before the trigger word (so "To avoid danger of
    suffocation" is removed completely, not only from "suffocation") and ends at the end of the
    sentence ("Contains: milk, soy." - the allergen list is not part of the ingredient list).
    """
    spans = []
    for m in STATEMENT_START_RE.finditer(text):
        separators = list(STATEMENT_BOUNDARY_RE.finditer(text, 0, m.start()))
        start = separators[-1].end() if separators else 0
        end = re.search(r"\.(\s|$)", text[m.end():])
        spans.append((start, m.end() + end.start() if end else len(text)))
    return spans


def _inside(start, end, spans) -> bool:
    return any(s <= start and end <= e for s, e in spans)


def _entity(text, start, end, label, rule, **extra):
    return {"start": start, "end": end, "text": text[start:end], "label": label, "rule": rule, **extra}


def _clean_piece(text: str, start: int, end: int):
    """Strip framing words and trailing percentages from text[start:end]; return new (start, end)."""
    while True:
        piece = text[start:end]
        m = FRAMING_PREFIX_RE.match(piece)
        if not m or m.end() == 0:
            break
        start += m.end()
    piece = text[start:end]
    m = TRAILING_NOISE_RE.search(piece)
    if m:
        end = start + m.start()
    piece = text[start:end]
    left = len(piece) - len(piece.lstrip(" ,.;:*-&"))
    right = len(piece.rstrip(" ,.;:*-&"))
    return start + left, start + right


def _lookup(phrase: str):
    return load_lexicon().get(lookup_key(phrase))


def _from_lexicon(text, start, end, entry, rule_prefix):
    extra = {}
    if entry.number:
        extra["number"] = entry.number
    if entry.class_id:
        extra["class_id"] = entry.class_id
    return _entity(text, start, end, entry.label, f"{rule_prefix}:{entry.source.split(':')[1]}", **extra)


def _word_spans(text, start, end):
    return [(m.start() + start, m.end() + start) for m in WORD_RE.finditer(text[start:end])]


def _scan_lexicon_terms(text, start, end):
    """Greedy longest-match search for dictionary terms inside text[start:end] (word-aligned)."""
    words = _word_spans(text, start, end)
    found, i = [], 0
    while i < len(words):
        for j in range(min(len(words), i + 7), i, -1):        # longest first, up to 7 words
            entry = _lookup(text[words[i][0]:words[j - 1][1]])
            if entry:
                found.append(_from_lexicon(text, words[i][0], words[j - 1][1], entry, "dict_span"))
                i = j
                break
        else:
            i += 1
    return found


def classify_piece(text: str, start: int, end: int, whole_list: bool = False) -> list:
    """Label one cleaned piece of a segment. Returns a list of entities (possibly empty)."""
    piece = text[start:end]
    if not piece or SKIP_PIECE_RE.match(piece) or not re.search(r"[^\W\d_]", piece):
        return []

    # 3. whole piece is a dictionary term
    entry = _lookup(piece)
    if entry:
        return [_from_lexicon(text, start, end, entry, "dict")]

    # 4. several terms joined by connectors: every part must be a dictionary term or match a head rule
    parts, pos = [], start
    for m in CONNECTOR_RE.finditer(piece):
        parts.append((pos, start + m.start()))
        pos = start + m.end()
    parts.append((pos, end))
    if len(parts) > 1:
        entities = []
        for s, e in parts:
            s, e = _clean_piece(text, s, e)
            part_entry = _lookup(text[s:e])
            words = [w.lower() for w in WORD_RE.findall(text[s:e])]
            rule = head_rule(words)
            if part_entry:
                entities.append(_from_lexicon(text, s, e, part_entry, "dict_conj"))
            elif rule and rule[0] != "INGREDIENT":
                entities.append(_entity(text, s, e, rule[0], rule[1] + "_conj"))
            else:
                entities = None
                break
        if entities:
            return entities

    words = _word_spans(text, start, end)
    lowered = [text[s:e].lower() for s, e in words]
    rule = head_rule(lowered)

    # exceptions to the dictionary that must win: "peanut butter" (not FAT), "spearmint oil" (FLAVOURING)
    if rule and rule[1] in ("head:nut_butter", "head:flavour_oil"):
        return [_entity(text, start, end, rule[0], rule[1])]

    # 5. two or more dictionary terms that together cover every content word:
    #    "Emulsifier soy lecithin" -> FUNCTION_CLASS + ADDITIVE
    found = _scan_lexicon_terms(text, start, end)
    if len(found) >= 2:
        covered = {w for w in words if any(f["start"] <= w[0] and w[1] <= f["end"] for f in found)}
        leftover = [text[s:e].lower() for s, e in words if (s, e) not in covered]
        if all(w in STOPWORDS for w in leftover):
            return found

    # 6. the piece ENDS with a dictionary term (longest suffix) or has a head word:
    #    "organic cane sugar" -> SUGAR, "refined palm oil" -> FAT
    for k in range(1, len(words)):                            # k = first word of the suffix
        suffix_entry = _lookup(text[words[k][0]:end])
        if not suffix_entry:
            continue
        prefix = lowered[:k]
        if suffix_entry.label in ADDITIVE_LABELS:
            absorb = all(w in ADDITIVE_MODIFIERS for w in prefix)
        else:
            absorb = len(prefix) <= SUFFIX_MAX_PREFIX_WORDS and all(w.isalpha() for w in prefix)
        if absorb:                                            # "organic cane sugar" -> one SUGAR span
            return [_from_lexicon(text, start, end, suffix_entry, "dict_suffix")]
        # "coconut with sodium metabisulfite" -> label the prefix separately + the dictionary term
        p_start, p_end = _clean_piece(text, start, words[k][0])
        return (classify_piece(text, p_start, p_end) if p_end > p_start else []) + \
            [_from_lexicon(text, words[k][0], end, suffix_entry, "dict_span")]
    if rule:
        return [_entity(text, start, end, rule[0], rule[1])]

    # 7. fallback: a short Latin-script piece made of words seen in other products is an ingredient
    if (len(words) <= MAX_FALLBACK_WORDS and piece[0].isalpha() and non_latin_letter_ratio(piece) == 0
            and (whole_list or _known_words(piece))):
        return [_entity(text, start, end, "INGREDIENT", "fallback:segment")]
    return []


def label_processed(processed: dict) -> list:
    """Weak-label the output of process_ingredient_text(). Returns entities sorted by position."""
    text = processed["text"]
    statements = find_statement_spans(text)
    entities = []

    # 1. additive codes
    codes = [c for c in processed["additive_codes"] if not _inside(c["start"], c["end"], statements)]
    for c in codes:
        entities.append(_entity(text, c["start"], c["end"], "INS_CODE", f"regex:ins_code_{c['style']}",
                                number=c["number"]))

    # 2. segments -> pieces between codes -> classify
    for seg in processed["segments"]:
        if _inside(seg["start"], seg["end"], statements):
            continue
        cut_points = [(c["start"], c["end"]) for c in codes if seg["start"] <= c["start"] < seg["end"]]
        pos, pieces = seg["start"], []
        for s, e in cut_points:
            pieces.append((pos, s))
            pos = e
        pieces.append((pos, seg["end"]))
        for s, e in pieces:
            # a statement can also start in the middle of a segment ("salt. Contains milk")
            stmt = next((st for st, en in statements if s <= st < e), None)
            if stmt is not None:
                e = stmt
            s, e = _clean_piece(text, s, e)
            if e > s:
                # a list that is ONE short segment ("Bhendi", "Raisins.") is the ingredient itself,
                # so the vocabulary check of the fallback rule is not needed
                whole_list = len(processed["segments"]) == 1
                entities.extend(classify_piece(text, s, e, whole_list=whole_list))

    return sorted(entities, key=lambda ent: ent["start"])


def label_text(text) -> dict:
    """Convenience wrapper: raw text -> processed dict + 'entities'."""
    processed = process_ingredient_text(text)
    processed["entities"] = label_processed(processed)
    return processed
