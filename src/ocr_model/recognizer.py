"""Our CRNN as the app's text recogniser: RapidOCR's detector finds the text lines, our model reads them.

    from src.ocr.ocr_pipeline import read_image
    read_image("photo.jpg", recognizer="crnn")

The model in models/ocr-crnn/model.pt is the tuned CRNN (RMSprop, batch 64; optimizer_study.py) fine-tuned on
all real line crops we have (377 packet-photo crops + Open Food Facts crops). Its test numbers come from the
two-fold evaluation (improve_crnn.py, optimizer_study.py), where every crop was read by a model that never saw
its photo: 8.7% character error rate on real crops (RapidOCR's recogniser: 6.9%).
Decoding: CTC beam search with our food character language model (alpha 0.7, beta 1.0 as chosen on validation).
"""
from functools import lru_cache

import cv2
import numpy as np
import torch

from src.ocr_model.models import build
from src.ocr_model.real_photos import pad_width
from src.ocr_model.render_lines import HEIGHT
from src.utils.config import project_path

MODEL_PATH = project_path("models/ocr-crnn/model.pt")
NAME = "Our CRNN (trained by us) + RapidOCR text detector"


def available() -> bool:
    return MODEL_PATH.exists()


@lru_cache(maxsize=1)
def _model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model = build(ckpt["model"]).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    return model, device, ckpt.get("beam", {"alpha": 0.7, "beta": 1.0})


def line_image(image: np.ndarray, quad) -> np.ndarray:
    """Cut a detected box out of the (colour) photo as a straight grey line 32 px high. Tall boxes (text running
    up or down) are turned the way RapidOCR does for its own recogniser."""
    q = np.asarray(quad, np.float32)
    w = int(round(max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))))
    h = int(round(max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))))
    if w < 4 or h < 4:
        return None
    crop = cv2.warpPerspective(image, cv2.getPerspectiveTransform(q, np.float32([[0, 0], [w, 0], [w, h], [0, h]])),
                               (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if h / w >= 1.5:
        crop = np.rot90(crop)
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    new_w = max(4, round(grey.shape[1] * HEIGHT / grey.shape[0]))
    return cv2.resize(grey, (new_w, HEIGHT), interpolation=cv2.INTER_AREA if grey.shape[0] > HEIGHT else cv2.INTER_CUBIC)


@torch.no_grad()
def read_boxes(image: np.ndarray, boxes: list) -> tuple:
    """(texts, confidences) for the detector's boxes. Confidence = mean probability of the characters read."""
    from src.ocr_model.combine import resources
    from src.ocr_model.improve_crnn import beam_decode
    model, device, beam = _model()
    lm, _ = resources()
    texts, scores = [], []
    for quad in boxes:
        img = line_image(image, quad)
        if img is None:
            texts.append("")
            scores.append(0.0)
            continue
        x = torch.from_numpy(pad_width(img)).float().div(255).view(1, 1, HEIGHT, -1).to(device)
        lp = model(x)[:, 0].float().cpu().numpy()                   # T x classes, log-probabilities
        texts.append(beam_decode(lp, lm, beam["alpha"], beam["beta"]))
        best = lp.max(axis=1)
        chars = lp.argmax(axis=1) != 0
        scores.append(float(np.exp(best[chars]).mean()) if chars.any() else 0.0)
    return texts, scores
