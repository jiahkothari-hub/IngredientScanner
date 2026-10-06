"""OCR: packet photo -> text, with EasyOCR.

    from src.ocr.ocr_pipeline import read_image
    result = read_image("data/images/packets/8901058000290.jpg")
    result["text"]      # full text in reading order, one line per printed line
    result["lines"]     # [{"text", "confidence", "box"}, ...]

Steps:
  1. Pre-processing (OpenCV): convert to grey, enlarge small photos (small print needs ~30 px letters),
     and equalise local contrast (CLAHE) so faded print on shiny packets becomes readable.
  2. EasyOCR: a CRAFT text detector finds word boxes, a CRNN recogniser reads each box.
  3. Reading order: boxes are grouped into lines by their vertical position, then sorted left to right.
     EasyOCR returns boxes in an unreliable order, and ingredient lists must be read in order.
  4. If very little text is found, the photo is probably rotated: try 90/180/270 degrees, keep the best.

EasyOCR's models (~100 MB) are downloaded once into models/easyocr/ on first use.
"""
from functools import lru_cache

import cv2
import numpy as np

from src.utils.config import project_path

MODEL_DIR = project_path("models/easyocr")
TARGET_MIN_SIDE = 1400      # enlarge photos smaller than this
MIN_CONFIDENCE = 0.15       # drop boxes the recogniser itself is very unsure about
MIN_CHARACTERS = 40         # below this, try rotating the photo


@lru_cache(maxsize=1)
def get_reader():
    import easyocr
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return easyocr.Reader(["en"], gpu=False, model_storage_directory=str(MODEL_DIR), verbose=False)


def preprocess(image: np.ndarray) -> np.ndarray:
    """Grey scale, enlarge if small, CLAHE contrast equalisation."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    short_side = min(grey.shape[:2])
    if short_side < TARGET_MIN_SIDE:
        scale = TARGET_MIN_SIDE / short_side
        grey = cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(grey)


def group_into_lines(detections: list) -> list:
    """Sort word boxes into reading order: same line if vertical centres are within half a box height."""
    boxes = []
    for box, text, conf in detections:
        if conf < MIN_CONFIDENCE or not text.strip():
            continue
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        boxes.append({"text": text, "confidence": float(conf), "box": [[int(x), int(y)] for x, y in box],
                      "cy": (min(ys) + max(ys)) / 2, "h": max(ys) - min(ys), "x": min(xs)})
    boxes.sort(key=lambda b: b["cy"])
    lines, current = [], []
    for b in boxes:
        if current and abs(b["cy"] - np.mean([c["cy"] for c in current])) > 0.5 * np.median([c["h"] for c in current]):
            lines.append(current)
            current = []
        current.append(b)
    if current:
        lines.append(current)
    result = []
    for line in lines:
        line.sort(key=lambda b: b["x"])
        result.append({"text": " ".join(b["text"] for b in line),
                       "confidence": round(float(np.mean([b["confidence"] for b in line])), 3),
                       "box": [b["box"] for b in line]})
    return result


def _ocr(image: np.ndarray) -> list:
    detections = get_reader().readtext(image, detail=1, paragraph=False)
    return group_into_lines(detections)


def read_image(path_or_array) -> dict:
    """Run the full OCR pipeline on a file path, bytes or an image array (BGR)."""
    if isinstance(path_or_array, (str, bytes)) or hasattr(path_or_array, "__fspath__"):
        if isinstance(path_or_array, bytes):
            image = cv2.imdecode(np.frombuffer(path_or_array, np.uint8), cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(str(path_or_array))
    else:
        image = path_or_array
    if image is None:
        raise ValueError("could not read the image")
    prepared = preprocess(image)
    lines, rotation = _ocr(prepared), 0
    if sum(len(l["text"]) for l in lines) < MIN_CHARACTERS:
        for angle, code in ((90, cv2.ROTATE_90_CLOCKWISE), (180, cv2.ROTATE_180), (270, cv2.ROTATE_90_COUNTERCLOCKWISE)):
            rotated = _ocr(cv2.rotate(prepared, code))
            if sum(len(l["text"]) for l in rotated) > sum(len(l["text"]) for l in lines):
                lines, rotation = rotated, angle
    return {"text": "\n".join(l["text"] for l in lines), "lines": lines, "rotation": rotation,
            "mean_confidence": round(float(np.mean([l["confidence"] for l in lines])), 3) if lines else 0.0}
