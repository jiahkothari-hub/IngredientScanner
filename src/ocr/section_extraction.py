"""Find the ingredient list inside the full OCR text of a packet.

A photo of a packet also contains the brand, nutrition table, address, storage advice ... We want only
the part after "Ingredients:" and before the next section.

Strategy (first that works):
  1. keyword: find every "Ingredients" (tolerant to OCR errors: "lngredients", "INGREDIENTS", "Ingredlents"),
     cut each candidate at the first end marker (nutrition, allergen advice, storage, manufacturer, FSSAI ...)
     and keep the candidate that looks most like a list (commas; a colon right after the keyword).
     This skips sentences such as "allergens: see ingredients in bold".
  2. comma density: no keyword found -> take the block of consecutive lines with the most commas
     (ingredient lists are the most comma-dense text on a pack).
  3. full text: nothing better found -> use everything (the NER step ignores non-ingredient text
     reasonably well, but precision drops).
"""
import re

START_RE = re.compile(r"\b[il1|]ngr[eéc]d[il1]?[eéc]nts?\b\s*[:;.\-]?", re.IGNORECASE)
END_RE = re.compile(
    r"\b(?:nutrition(?:al)?(?:\s+(?:information|facts|value))?|typical\s+values|energy\s*\(?k?j|"
    r"allerg(?:en|y)\s+(?:information|advice)|storage|store\s+(?:in|at)|keep\s+(?:refrigerated|in)|"
    r"best\s+before|use\s+by|manufactured\s+(?:by|for)|marketed\s+by|packed\s+by|mfd\.?\s+by|"
    r"net\s+(?:wt|weight|qty|quantity)|mrp|m\.r\.p|[fj]s+a[il1]|lic\.?\s*no|customer\s+care|cust\.?\s+care|"
    r"directions|serving\s+suggestion|preparation|batch\s+no|www\.)",
    re.IGNORECASE,
)


def extract_ingredients_section(ocr_text: str) -> dict:
    """Return {"text", "method", "start", "end"} with offsets into `ocr_text`."""
    flat = ocr_text.replace("\n", " ")
    candidates = []
    for start in START_RE.finditer(flat):
        end = END_RE.search(flat, start.end())
        stop = end.start() if end else len(flat)
        text = flat[start.end():stop].strip(" :;.-")
        # a real ingredient list has separators; "Ingredients:" with a colon is a strong signal
        score = text.count(",") + text.count(";") + (3 if ":" in start.group(0) else 0)
        candidates.append((score, len(text), start.end(), stop, text))
    if candidates:
        score, _, begin, stop, text = max(candidates)
        if text:
            return {"text": text, "method": "keyword", "start": begin, "end": stop}

    lines = ocr_text.split("\n")
    best, best_score = None, 0
    for i in range(len(lines)):
        for j in range(i + 1, min(len(lines), i + 15) + 1):
            block = " ".join(lines[i:j])
            score = block.count(",") - 0.3 * (j - i)          # many commas, few extra lines
            if score > best_score:
                best, best_score = (i, j), score
    if best and best_score >= 3:
        text = " ".join(lines[best[0]:best[1]])
        start_offset = len(" ".join(lines[:best[0]])) + (1 if best[0] else 0)
        return {"text": text, "method": "comma_density", "start": start_offset, "end": start_offset + len(text)}
    return {"text": flat.strip(), "method": "full_text", "start": 0, "end": len(flat)}
