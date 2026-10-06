"""OCR: packet photo -> text.

    from src.ocr.ocr_pipeline import read_image
    result = read_image("data/images/packets/8901058000290.jpg")
    result["text"]        # text in reading order, one line per printed line (spelling-corrected)
    result["raw_text"]    # the same before spelling correction
    result["lines"]       # [{"text", "confidence", "box"}, ...]

Engine: RapidOCR 3 = the PaddleOCR PP-OCRv6 text detector + recogniser exported to ONNX (runs on any CPU;
the models ship inside the pip package, no download). On small packet print it reads whole sentences
with correct spaces and punctuation, where EasyOCR merged or dropped words; it is also ~5x faster.
EasyOCR is kept as a fallback when RapidOCR is not installed.

Steps:
  1. Load + fix orientation: phone photos store their rotation in EXIF; we apply it first. A slightly
     tilted photo (1-20 degrees, measured from the detected text lines) is straightened.
  2. Adaptive enlargement: ingredient lists are the SMALLEST print on a pack. A fast detection-only pass
     measures the height of the small text lines; the photo is then enlarged so that this print is
     ~36 px high (the size the recogniser works best at). Huge photos are reduced first.
  3. Detection + recognition on the enlarged photo. No confidence filter: a low-confidence line is
     usually a real, slightly blurred ingredient line, and the later steps tolerate errors.
  4. Reading order: a recursive XY-cut finds text blocks and columns (labels often print the ingredient
     list next to the importer address); inside each block, boxes are grouped into lines and sorted left
     to right. Tables (rows aligned across the gap) are kept together so "Sugars ... 12 g" stays one line.
     Prose columns that touch (no empty channel) are recognised as side-by-side left-aligned paragraphs.
  5. Orientation: phone photos are often sideways or upside down. Sideways text is easy to spot: the
     detector's text-line boxes are taller than wide. Then a small copy is read at 90 and 270 degrees;
     if the upright reading is poor for another reason, at all four angles. The angle with the most
     confidently read text wins (the per-line flip classifier is off, so an upside-down reading really
     scores low instead of being read in reverse line order).
  6. Spelling correction with a food-domain dictionary (src/ocr/text_correction.py).
"""
import re
from functools import lru_cache

import cv2
import numpy as np

from src.utils.config import project_path

MODEL_DIR = project_path("models/easyocr")
TARGET_TEXT_HEIGHT = 36       # px height of the small print after enlargement
MAX_SCALE = 4.0
MAX_PIXELS = 14_000_000       # memory/time guard for the enlarged photo
MAX_INPUT_SIDE = 3000         # very large phone photos are reduced to this first
MIN_CHARACTERS = 40           # below this, try rotating the photo


# ------------------------------------------------------------------------------------------- engines
@lru_cache(maxsize=1)
def get_rapidocr():
    try:
        from rapidocr import RapidOCR
    except ImportError:
        return None
    return RapidOCR(params={"Global.log_level": "warning"})


@lru_cache(maxsize=1)
def get_reader():
    """EasyOCR fallback (only used when RapidOCR is not installed)."""
    import easyocr
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return easyocr.Reader(["en"], gpu=False, model_storage_directory=str(MODEL_DIR), verbose=False)


def engine_name() -> str:
    return "RapidOCR (PP-OCRv6)" if get_rapidocr() is not None else "EasyOCR"


# ------------------------------------------------------------------------------------------- image
def load_image(path_or_array) -> np.ndarray:
    """File path, bytes or BGR array -> BGR array, with the EXIF rotation of phone photos applied."""
    if isinstance(path_or_array, np.ndarray):
        return path_or_array
    from io import BytesIO
    from PIL import Image, ImageOps
    if isinstance(path_or_array, (bytes, bytearray)):
        pil = Image.open(BytesIO(path_or_array))
    else:
        pil = Image.open(str(path_or_array))
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _detect(image: np.ndarray) -> list:
    engine = get_rapidocr()
    if engine is None:
        return []
    boxes = engine(image, use_det=True, use_cls=False, use_rec=False).boxes
    return [] if boxes is None else list(boxes)


def vertical_fraction(boxes) -> float:
    """Share of text boxes (weighted by size) that are much taller than wide = text running up/down."""
    if not len(boxes):
        return 0.0
    vertical = total = 0.0
    for b in boxes:
        b = np.asarray(b, dtype=float)
        width = np.linalg.norm(b[1] - b[0])
        height = np.linalg.norm(b[2] - b[1])
        weight = width * height
        total += weight
        if height > 1.5 * width:
            vertical += weight
    return vertical / total if total else 0.0


def skew_angle(boxes) -> float:
    """Median tilt (degrees) of the horizontal text lines, weighted by their length."""
    angles, weights = [], []
    for b in boxes:
        b = np.asarray(b, dtype=float)
        dx, dy = b[1] - b[0]
        if abs(dx) > 2 * abs(dy) and abs(dx) > 20:          # horizontal-ish line boxes only
            angles.append(np.degrees(np.arctan2(dy, dx)))
            weights.append(abs(dx))
    if len(angles) < 3:
        return 0.0
    order = np.argsort(angles)
    cumulative = np.cumsum(np.asarray(weights)[order])
    return float(np.asarray(angles)[order][np.searchsorted(cumulative, cumulative[-1] / 2)])


def deskew(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate by `angle` degrees around the centre, enlarging the canvas so nothing is cut off."""
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    matrix[0, 2] += new_w / 2 - w / 2
    matrix[1, 2] += new_h / 2 - h / 2
    return cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def choose_scale(image: np.ndarray, boxes=None) -> float:
    """Scale factor so that the small print (25th percentile of text-line heights) becomes ~36 px."""
    h, w = image.shape[:2]
    boxes = _detect(image) if boxes is None else boxes
    small_text = None
    heights = [min(np.linalg.norm(np.asarray(b[1], float) - np.asarray(b[0], float)),
                   np.linalg.norm(np.asarray(b[2], float) - np.asarray(b[1], float))) for b in boxes]
    heights = [x for x in heights if x >= 4]
    if heights:
        small_text = float(np.percentile(heights, 25))
    if small_text is None:                       # nothing detected: enlarge small photos moderately
        scale = 1400 / min(h, w)
    else:
        scale = TARGET_TEXT_HEIGHT / small_text
    scale = min(scale, MAX_SCALE, (MAX_PIXELS / (h * w)) ** 0.5)
    return max(scale, 1.0)


def preprocess(image: np.ndarray, scale: float) -> np.ndarray:
    if scale > 1.01:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return image


# ------------------------------------------------------------------------------------------- lines
def _rows(boxes: list) -> list:
    """Group boxes of ONE text block into printed lines: same line if the vertical centres are within half
    a box height and the boxes do not overlap horizontally (overlapping boxes are on different lines)."""
    lines = []
    for b in sorted(boxes, key=lambda b: b["cy"]):
        for line in reversed(lines[-3:]):
            close = abs(b["cy"] - np.mean([c["cy"] for c in line])) <= 0.5 * np.median([c["h"] for c in line])
            overlap = any(min(b["x1"], c["x1"]) - max(b["x0"], c["x0"]) > 0.3 * min(b["x1"] - b["x0"], c["x1"] - c["x0"])
                          for c in line)
            if close and not overlap:
                line.append(b)
                break
        else:
            lines.append([b])
    lines.sort(key=lambda line: np.mean([c["cy"] for c in line]))
    return [sorted(line, key=lambda c: c["x0"]) for line in lines]


def _gaps(intervals: list, min_gap: float) -> list:
    """Split points of a set of 1-D intervals at empty gaps wider than min_gap."""
    intervals = sorted(intervals)
    groups, current, end = [], [intervals[0]], intervals[0][1]
    for iv in intervals[1:]:
        if iv[0] > end + min_gap:
            groups.append(current)
            current = []
        current.append(iv)
        end = max(end, iv[1])
    groups.append(current)
    return groups


NUMERIC_CELL_RE = re.compile(r"[\d\s.,%()*<>/~:-]*\d[\d\s.,%()*<>/~:-]*(?:\s*(?:k?cal|kj|mg|µg|μg|mcg|g|ml|iu))?"
                             r"(?:\s*[\d.,%()*/~-]*\s*(?:k?cal|kj|mg|µg|μg|mcg|g|ml|iu)?)*", re.IGNORECASE)


def _is_table(left: list, right: list, mh: float) -> bool:
    """Table check: most rows on the left have a box on the right at the same height, AND one side is
    mostly numbers ("2214 kcal", "0.9 g"). Two columns of prose with the same line spacing (ingredient
    list next to an address) are not a table."""
    matched = sum(any(abs(a["cy"] - b["cy"]) < 0.35 * mh for b in right) for a in left)
    if matched < 0.5 * len(left):
        return False
    numeric = lambda col: np.mean([bool(NUMERIC_CELL_RE.fullmatch(b["text"].strip())) for b in col])
    return max(numeric(left), numeric(right)) >= 0.5


def _xy_cut(boxes: list, depth: int = 0) -> list:
    """Reading order for a label with several text blocks/columns (recursive XY-cut): split at empty
    horizontal bands, then at empty vertical channels (columns, read left to right). A vertical split is
    NOT made when the rows on both sides line up - that is a table (nutrient name ... value)."""
    if len(boxes) <= 1 or depth > 8:
        return _rows(boxes)
    mh = float(np.median([b["h"] for b in boxes]))
    bands = _gaps([(b["y0"], b["y1"], i) for i, b in enumerate(boxes)], 0.6 * mh)
    if len(bands) > 1:
        return [line for band in bands for line in _xy_cut([boxes[i] for _, _, i in band], depth + 1)]
    columns = _gaps([(b["x0"], b["x1"], i) for i, b in enumerate(boxes)], 1.0 * mh)
    if len(columns) > 1:
        cols = [[boxes[i] for _, _, i in col] for col in columns]
        if not any(_is_table(a, b, mh) for a, b in zip(cols, cols[1:])):
            return [line for col in cols for line in _xy_cut(col, depth + 1)]
    return _rows(boxes)


def _aligned_blocks(boxes: list) -> list:
    """Chain boxes into left-aligned paragraphs: a box continues a paragraph if it starts at the same x
    as the paragraph's last line and lies one line below it."""
    mh = float(np.median([b["h"] for b in boxes]))
    blocks = []
    for b in sorted(boxes, key=lambda b: b["y0"]):
        best, best_dy = None, None
        for block in blocks:
            last = block[-1]
            dy = b["y0"] - last["y0"]
            if abs(b["x0"] - last["x0"]) < 0.8 * mh and 0.3 * mh < dy < 2.2 * mh \
                    and 0.6 < b["h"] / max(1, last["h"]) < 1.6 and (best_dy is None or dy < best_dy):
                best, best_dy = block, dy
        if best is None:
            blocks.append([b])
        else:
            best.append(b)
    return blocks


def _untangle_columns(lines: list, boxes: list) -> list:
    """Side-by-side prose columns whose edges touch (no empty channel for the XY-cut) end up interleaved
    line by line ("...SUGAR, NON-HYDROGENATED Imported by: Mondelez..."). If two or more left-aligned
    paragraphs of 3+ text lines (not numbers) run side by side, each paragraph is read completely, in
    order, at the place where it starts."""
    prose = [blk for blk in _aligned_blocks(boxes) if len(blk) >= 3
             and np.mean([bool(NUMERIC_CELL_RE.fullmatch(b["text"].strip())) for b in blk]) < 0.5]
    side_by_side = set()
    for i, a in enumerate(prose):
        for j, c in enumerate(prose[:i]):
            overlap = min(a[-1]["y1"], c[-1]["y1"]) - max(a[0]["y0"], c[0]["y0"])
            span = min(a[-1]["y1"] - a[0]["y0"], c[-1]["y1"] - c[0]["y0"])
            if overlap > 0.5 * span:
                side_by_side.update((i, j))
    if not side_by_side:
        return lines
    block_of = {id(b): k for k in side_by_side for b in prose[k]}
    out, emitted = [], set()
    for line in lines:
        rest = [b for b in line if id(b) not in block_of]
        for b in line:
            k = block_of.get(id(b))
            if k is not None and k not in emitted:
                emitted.add(k)
                out.extend([[x] for x in sorted(prose[k], key=lambda x: x["y0"])])
        if rest:
            out.append(rest)
    return out


def group_into_lines(detections: list) -> list:
    """Put word/line boxes into reading order (columns first, then lines, then left to right).
    detections = [(box, text, confidence), ...]."""
    boxes = []
    for box, text, conf in detections:
        if not str(text).strip():
            continue
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        boxes.append({"text": str(text).strip(), "confidence": float(conf),
                      "box": [[int(x), int(y)] for x, y in box], "x0": min(xs), "x1": max(xs),
                      "y0": min(ys), "y1": max(ys), "cy": (min(ys) + max(ys)) / 2, "h": max(ys) - min(ys)})
    if not boxes:
        return []
    return [{"text": " ".join(b["text"] for b in line),
             "confidence": round(float(np.mean([b["confidence"] for b in line])), 3),
             "box": [b["box"] for b in line]} for line in _untangle_columns(_xy_cut(boxes), boxes)]


def _ocr(image: np.ndarray, unclip_ratio: float = None) -> list:
    engine = get_rapidocr()
    if engine is not None:
        out = engine(image, use_det=True, use_cls=False, use_rec=True, text_score=0.3, unclip_ratio=unclip_ratio)
        if out.boxes is None or out.txts is None:
            return []
        return group_into_lines(list(zip(out.boxes.tolist(), out.txts, out.scores)))
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grey = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grey)
    return group_into_lines(get_reader().readtext(grey, detail=1, paragraph=False))


ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
PROBE_SIDE = 1100


def rotate(image: np.ndarray, angle: int) -> np.ndarray:
    return image if not angle else cv2.rotate(image, ROTATIONS[angle])


def best_orientation(image: np.ndarray, candidates) -> int:
    """Which of the candidate angles reads best on a small copy of the photo."""
    h, w = image.shape[:2]
    f = min(1.0, PROBE_SIDE / max(h, w))
    small = cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else image
    scores = {angle: _quality(_ocr(rotate(small, angle))) for angle in candidates}
    best = max(scores, key=scores.get)
    if 0 in scores and scores[best] < 1.3 * scores[0]:      # turning must be clearly better than not turning
        return 0
    return best


def _mean_confidence(lines) -> float:
    return float(np.mean([l["confidence"] for l in lines])) if lines else 0.0


def _best_reading(image: np.ndarray) -> list:
    """Read the photo; if the reading is unsure, read it again with tighter text boxes and keep the
    better one. (In dense paragraphs with little line spacing, the detector's default box expansion
    merges two printed lines into one box, which the recogniser cannot read.)"""
    lines = _ocr(image)
    if get_rapidocr() is not None and _mean_confidence(lines) < 0.9:
        tight = _ocr(image, unclip_ratio=1.0)
        if _quality(tight) > _quality(lines):
            lines = tight
    return lines


def _n_chars(lines) -> int:
    return sum(len(l["text"]) for l in lines)


def _is_horizontal(box) -> bool:
    b = np.asarray(box, dtype=float)
    return np.linalg.norm(b[1] - b[0]) >= np.linalg.norm(b[2] - b[1])


def _quality(lines) -> float:
    """Amount of confidently read text in HORIZONTAL lines: characters weighted by the recogniser's
    confidence. Vertical boxes do not count: the recogniser turns tall crops itself, so a sideways photo
    can be read word by word, but then the line order is wrong."""
    return sum(len(l["text"]) * l["confidence"] for l in lines if all(_is_horizontal(b) for b in l["box"]))


def read_image(path_or_array, correct_spelling: bool = True) -> dict:
    """Run the full OCR pipeline on a file path, bytes or an image array (BGR)."""
    try:
        image = load_image(path_or_array)
    except Exception as err:                       # not an image / corrupt file
        raise ValueError(f"could not read the image ({err})") from err
    if image is None or image.size == 0:
        raise ValueError("could not read the image")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape[:2]
    if max(h, w) > MAX_INPUT_SIDE:
        f = MAX_INPUT_SIDE / max(h, w)
        image = cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)

    boxes = _detect(image)
    angle = skew_angle(boxes)
    if 1.0 <= abs(angle) <= 20:                                 # slightly tilted photo: straighten it
        image = deskew(image, angle)
        boxes = _detect(image)
    if vertical_fraction(boxes) >= 0.5:                       # sideways photo
        rotation = best_orientation(image, (90, 270))
    else:
        rotation = 0
        scale = choose_scale(image, boxes)
        lines = _best_reading(preprocess(image, scale))
        if _n_chars(lines) < 150 or _mean_confidence(lines) < 0.85:    # poor reading: upside down, or
            rotation = best_orientation(image, (0, 90, 180, 270))        # sideways text the detector missed
    if rotation:
        image = rotate(image, rotation)
        scale = choose_scale(image)
        lines = _best_reading(preprocess(image, scale))

    raw_text = "\n".join(l["text"] for l in lines)
    text = raw_text
    if correct_spelling and raw_text:
        from src.ocr.text_correction import correct_ocr_text
        text = correct_ocr_text(raw_text)
    return {"text": text, "raw_text": raw_text, "lines": lines, "rotation": rotation, "scale": round(scale, 2),
            "engine": engine_name(),
            "mean_confidence": round(_mean_confidence(lines), 3)}
