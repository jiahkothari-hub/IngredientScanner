"""Synthetic text-line images made from our real ingredient lists (unlimited, exactly labelled training data).

    python -m src.ocr_model.render_lines --samples 24                 # sample sheet of training images
    python -m src.ocr_model.render_lines --samples 24 --split test --severity 2

Every image is made from (split, seed, index) alone, so a dataset is reproducible without storing images.
  * Text:  a line cut from a silver-split ingredient list of the same split, so lines of test products are
           never seen in training.
  * Fonts: the test fonts (TEST_FONTS) are never used for training, so the test measures whether the model
           reads letter shapes it has not seen, not whether it memorised a font.
  * Look:  label-like printing (dark on light or light on dark, tracking, bold, italic, neighbouring lines cut
           off at the edge, rules), then phone-photo damage: rotation, perspective, packet curvature, blur,
           motion blur, shadow, glare, low resolution, sensor noise and JPEG compression.
  * Severity 0 = clean print, 1 = typical phone photo (training data), 2 = hard photo (robustness test).
  * tough=True adds what the real-photo test showed the first CRNN could not read: condensed type, mixed
    bold/regular in one line, faded low-contrast print, white print on mid-grey, halftone dots, uneven light
    and stronger blur.
"""
import argparse
import gzip
import json
import os
import random
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.ocr_model.charset import clean_text, encode
from src.utils.config import project_path

HEIGHT = 32           # every line image is scaled to this height
STRIDE = 4            # every model outputs one time step per 4 pixel columns
MAX_WIDTH = 800
MAX_CHARS = 48

FONT_DIRS = [Path(os.environ.get("OCR_FONT_DIR", "fonts")), Path("C:/Windows/Fonts"),
             Path("/usr/share/fonts/truetype"), Path("/Library/Fonts")]
TRAIN_FONTS = ["arial.ttf", "arialbd.ttf", "ARIALN.TTF", "ARIALNB.TTF", "calibri.ttf", "calibrib.ttf",
               "Candara.ttf", "Candarab.ttf", "corbel.ttf", "corbelb.ttf", "consola.ttf", "cour.ttf", "courbd.ttf",
               "tahoma.ttf", "tahomabd.ttf", "times.ttf", "timesbd.ttf", "georgia.ttf", "segoeui.ttf",
               "segoeuib.ttf", "bahnschrift.ttf", "BOOKOS.TTF", "CENTURY.TTF", "framd.ttf", "pala.ttf",
               "constan.ttf", "lucon.ttf", "micross.ttf", "Nirmala.ttf", "LeelawUI.ttf", "ebrima.ttf"]
TEST_FONTS = ["verdana.ttf", "verdanab.ttf", "trebuc.ttf", "trebucbd.ttf", "GIL_____.TTF", "GILB____.TTF",
              "seguisb.ttf", "ROCK.TTF", "FRAMDCN.TTF", "palab.ttf"]


# ---------------------------------------------------------------------------------------- inputs
def _find(name: str):
    for folder in FONT_DIRS:
        path = folder / name
        if path.exists():
            return str(path)
    return None


@lru_cache(maxsize=None)
def font_paths(split: str) -> tuple:
    """Fonts for a split: TEST_FONTS for 'test', TRAIN_FONTS otherwise (only those installed). On a machine
    without these fonts, every .ttf found is split 80/20 by name instead."""
    paths = [p for p in map(_find, TEST_FONTS if split == "test" else TRAIN_FONTS) if p]
    if len(paths) >= 3:
        return tuple(paths)
    found = sorted({str(p) for d in FONT_DIRS if d.exists() for p in d.rglob("*.ttf")})
    if not found:
        raise FileNotFoundError("no .ttf fonts found: set OCR_FONT_DIR to a folder of fonts")
    cut = max(1, len(found) // 5)
    return tuple(found[:cut] if split == "test" else found[cut:])


@lru_cache(maxsize=None)
def split_texts(split: str) -> tuple:
    """Cleaned ingredient lists of one silver split ('train', 'validation' or 'test')."""
    with gzip.open(project_path(f"data/splits/silver/{split}.jsonl.gz"), "rt", encoding="utf-8") as f:
        texts = [clean_text(json.loads(line)["text"]) for line in f]
    return tuple(t for t in texts if len(t) >= 8)


@lru_cache(maxsize=256)
def _font(path: str, size: int):
    return ImageFont.truetype(path, size)


def sample_line(texts, rng: random.Random) -> str:
    """A label-like line: a run of whole words (6-40 characters) from a random list, sometimes in capitals."""
    words = rng.choice(texts).split()
    start = rng.randrange(len(words))
    target = rng.randint(6, 40)
    line = words[start]
    for word in words[start + 1:]:
        if len(line) + 1 + len(word) > target:
            break
        line += " " + word
    line = line[:MAX_CHARS].strip()
    r = rng.random()
    if r < 0.25:
        line = line.upper()
    elif r < 0.30:
        line = line.title()
    return line


# ---------------------------------------------------------------------------------------- rendering
def _background(h: int, w: int, value: float, rng: random.Random) -> np.ndarray:
    """Base tone + a gentle gradient + low-frequency texture (printed film, cardboard)."""
    bg = np.full((h, w), value, np.float32)
    if rng.random() < 0.6:
        ramp = np.linspace(-1, 1, w if rng.random() < 0.5 else h, dtype=np.float32) * rng.uniform(5, 30)
        bg += ramp[None, :] if ramp.size == w else ramp[:, None]
    if rng.random() < 0.5:
        small = np.random.default_rng(rng.randrange(2**31)).normal(0, rng.uniform(3, 12), (max(2, h // 8), max(2, w // 8)))
        bg += cv2.resize(small.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    return bg


def _text_mask(text, font, tracking, stroke, size):
    """White-on-black mask of the text and its width. Drawn character by character when tracking != 0."""
    ascent, descent = font.getmetrics()
    width = int(sum(font.getlength(ch) for ch in text) + tracking * (len(text) - 1)) if tracking else int(font.getlength(text))
    mask = Image.new("L", (max(width, 1) + 2 * stroke + 2, ascent + descent + 2 * stroke + 2), 0)
    draw = ImageDraw.Draw(mask)
    if tracking:
        x = stroke + 1
        for ch in text:
            draw.text((x, stroke + 1), ch, font=font, fill=255, stroke_width=stroke, stroke_fill=255)
            x += font.getlength(ch) + tracking
    else:
        draw.text((stroke + 1, stroke + 1), text, font=font, fill=255, stroke_width=stroke, stroke_fill=255)
    return np.asarray(mask, np.float32) / 255.0


def _join(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Two text masks side by side, bottom-aligned (same font size, different weight)."""
    h = max(left.shape[0], right.shape[0])
    pad = lambda m: np.pad(m, ((h - m.shape[0], 0), (0, 0)))
    return np.hstack([pad(left), pad(right)])


def _curve(img: np.ndarray, amplitude: float, rng: random.Random) -> np.ndarray:
    """Bend the line like text printed on a round packet (sinusoidal vertical shift)."""
    h, w = img.shape
    phase, periods = rng.uniform(0, np.pi), rng.uniform(0.3, 1.0)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    shift = amplitude * np.sin(phase + 2 * np.pi * periods * xs / w)
    return cv2.remap(img, xs, ys + shift, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _perspective(img: np.ndarray, amount: float, rng: random.Random) -> np.ndarray:
    h, w = img.shape
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = [(rng.uniform(-amount, amount) * h, rng.uniform(-amount, amount) * h) for _ in range(4)]
    dst = np.float32([[x + dx, y + dy] for (x, y), (dx, dy) in zip(src, jitter)])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REPLICATE)


def render(text: str, font_path: str, rng: random.Random, severity: int = 1, tough: bool = False) -> np.ndarray:
    """One grey text-line image (uint8, HEIGHT x width) showing exactly `text`.
    tough=True adds the print and photo effects our real packet photos showed the first CRNN could not read
    (see TOUGH below); tough=False gives the original training images."""
    size = rng.randint(30, 56)
    font = _font(font_path, size)
    tracking = rng.choice([0, 0, 0, 0, 0, 0, -1, 1, 2, 3]) * size / 30
    stroke = 1 if rng.random() < 0.12 else 0                       # extra-bold print
    if tough and " " in text and rng.random() < 0.15:              # mixed weight: "(**wheat** flour, ..."
        cut = rng.choice([i for i, ch in enumerate(text) if ch == " "])
        mask = _join(_text_mask(text[:cut + 1], font, tracking, rng.randint(1, 2), size),
                     _text_mask(text[cut + 1:], font, tracking, 0, size))
    else:
        mask = _text_mask(text, font, tracking, stroke, size)
    if tough and rng.random() < 0.3:                                # condensed type: squeeze horizontally
        mask = cv2.resize(mask, (max(1, round(mask.shape[1] * rng.uniform(0.6, 0.95))), mask.shape[0]),
                          interpolation=cv2.INTER_AREA)

    if rng.random() < 0.12:                                         # italic: shear the mask
        shear = rng.uniform(0.12, 0.25)
        h, w = mask.shape
        mask = cv2.warpAffine(mask, np.float32([[1, -shear, shear * h], [0, 1, 0]]), (w + int(shear * h) + 1, h))

    # margins, then neighbouring lines / a rule cut off at the edges (they are not part of the label)
    h, w = mask.shape
    top, bottom = int(rng.uniform(0.04, 0.3) * h), int(rng.uniform(0.04, 0.3) * h)
    left, right = int(rng.uniform(0.02, 0.5) * size), int(rng.uniform(0.02, 0.5) * size)
    canvas = np.zeros((h + top + bottom, w + left + right), np.float32)
    canvas[top:top + h, left:left + w] = mask
    if severity >= 1 and rng.random() < 0.3:
        other = _text_mask(clean_text("".join(rng.choice("abcdefghilmnorstu ") for _ in range(len(text)))), font, 0, 0, size)
        oh, ow = other.shape
        overlap = int(rng.uniform(0.2, 0.45) * oh)
        ow = min(ow, canvas.shape[1])
        if rng.random() < 0.5:
            canvas[:overlap, :ow] = np.maximum(canvas[:overlap, :ow], other[oh - overlap:, :ow])
        else:
            canvas[-overlap:, :ow] = np.maximum(canvas[-overlap:, :ow], other[:overlap, :ow])
    if severity >= 1 and rng.random() < 0.08:
        y = rng.choice([1, canvas.shape[0] - 2])
        canvas[y:y + max(1, size // 20), :] = 1.0

    # colours: dark print on a light pack (most labels) or light print on a dark pack
    if rng.random() < 0.2:
        bg_value, fg_value = rng.uniform(15, 100), rng.uniform(170, 255)
    else:
        bg_value, fg_value = rng.uniform(165, 255), rng.uniform(0, 95)
    if tough and severity >= 1:
        r = rng.random()
        if r < 0.2:                                                 # faded / low-contrast print
            bg_value = rng.uniform(120, 240)
            fg_value = bg_value + rng.choice([-1, 1]) * rng.uniform(35, 80)
        elif r < 0.3:                                               # white print on a mid-grey pack
            bg_value, fg_value = rng.uniform(90, 150), rng.uniform(200, 255)
        if rng.random() < 0.12:                                     # halftone screen: text made of dots
            period = rng.uniform(2.5, 5.0)
            yy, xx = np.mgrid[:canvas.shape[0], :canvas.shape[1]].astype(np.float32)
            dots = (np.sin(2 * np.pi * xx / period) * np.sin(2 * np.pi * yy / period) + 1) / 2
            canvas = canvas * (rng.uniform(0.3, 0.6) + rng.uniform(0.4, 0.7) * dots)
    if severity == 0:
        bg_value, fg_value = 245.0, 15.0
    img = _background(*canvas.shape, bg_value, rng) if severity else np.full(canvas.shape, bg_value, np.float32)
    img = img * (1 - canvas) + fg_value * canvas

    if severity >= 1:
        hard = severity >= 2
        img = cv2.warpAffine(img, cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2),
                             rng.uniform(-1, 1) * (3 if hard else 1.5), 1.0), img.shape[::-1], borderMode=cv2.BORDER_REPLICATE)
        if rng.random() < 0.5:
            img = _perspective(img, 0.12 if hard else 0.06, rng)
        if rng.random() < 0.3:
            img = _curve(img, rng.uniform(0.02, 0.10 if hard else 0.06) * img.shape[0], rng)
        if rng.random() < 0.3:                                      # shadow across the line
            img *= np.linspace(rng.uniform(0.5, 1.0), rng.uniform(0.7, 1.0), img.shape[1], dtype=np.float32)[None, :]
        if rng.random() < (0.35 if hard else 0.12):                 # glare from shiny film
            yy, xx = np.mgrid[:img.shape[0], :img.shape[1]].astype(np.float32)
            cx, cy, r = rng.uniform(0, img.shape[1]), rng.uniform(0, img.shape[0]), rng.uniform(0.5, 2.0) * img.shape[0]
            img += rng.uniform(60, 160) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
        if tough and rng.random() < 0.3:                            # uneven light over the whole line
            small = np.random.default_rng(rng.randrange(2**31)).uniform(0.6, 1.15, (2, 4)).astype(np.float32)
            img *= cv2.resize(small, img.shape[::-1], interpolation=cv2.INTER_CUBIC)
        sigma = rng.uniform(0, 2.4 if hard else (2.2 if tough else 1.4))
        if sigma > 0.3:
            img = cv2.GaussianBlur(img, (0, 0), sigma)
        if rng.random() < (0.3 if hard else 0.15):                  # hand shake
            k = rng.randint(3, 9 if hard else 6)
            kernel = np.zeros((k, k), np.float32)
            kernel[k // 2, :] = 1.0 / k
            img = cv2.filter2D(img, -1, kernel)

    # scale to the model height (and, for photos, sometimes through an even lower resolution first)
    h, w = img.shape
    new_w = max(STRIDE, round(w * HEIGHT / h))
    if severity >= 1 and rng.random() < (0.6 if severity >= 2 else 0.3):
        low = rng.randint(14 if severity >= 2 else 18, 26)
        img = cv2.resize(img, (max(4, round(w * low / h)), low), interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, (new_w, HEIGHT), interpolation=cv2.INTER_AREA if h > HEIGHT else cv2.INTER_LINEAR)

    if severity >= 1:
        img += np.random.default_rng(rng.randrange(2**31)).normal(0, rng.uniform(0, 14 if severity >= 2 else 7), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if severity >= 1 and rng.random() < 0.4:
        quality = rng.randint(10, 50) if severity >= 2 else rng.randint(25, 90)
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1], cv2.IMREAD_GRAYSCALE)
    return fit_width(img, text)


def fit_width(img: np.ndarray, text: str) -> np.ndarray:
    """Make the width a multiple of STRIDE, wide enough for CTC (one step per character, plus a blank between
    repeated characters) and at most MAX_WIDTH."""
    repeats = sum(a == b for a, b in zip(text, text[1:]))
    need = STRIDE * (len(text) + repeats + 2)
    w = min(MAX_WIDTH, max(img.shape[1], need))
    w = -(-w // STRIDE) * STRIDE
    if w != img.shape[1]:
        img = cv2.resize(img, (w, HEIGHT), interpolation=cv2.INTER_LINEAR)
    return img


# ---------------------------------------------------------------------------------------- dataset
SPLIT_OFFSET = {"train": 0, "validation": 1, "test": 2}


class SyntheticLines(torch.utils.data.Dataset):
    """`size` reproducible synthetic lines of a split. Item = (1 x 32 x W float tensor in [0, 1], text)."""

    def __init__(self, split: str, size: int, seed: int = 0, severity: int = 1, tough: bool = False):
        self.split, self.size, self.seed, self.severity, self.tough = split, size, seed, severity, tough

    def __len__(self):
        return self.size

    def item(self, index: int):
        rng = random.Random(((self.seed * 3 + SPLIT_OFFSET[self.split]) * 1_000_003 + index) * 7919 + self.severity)
        text = sample_line(split_texts(self.split), rng)
        image = render(text, rng.choice(font_paths(self.split)), rng, self.severity, self.tough)
        return image, text

    def __getitem__(self, index: int):
        image, text = self.item(index)
        return torch.from_numpy(image).float().div_(255).unsqueeze(0), text


def collate(batch):
    """Pad images to the widest in the batch (with each image's own edge colour) and pack CTC targets."""
    width = max(img.shape[-1] for img, _ in batch)
    images = torch.empty(len(batch), 1, HEIGHT, width)
    for i, (img, _) in enumerate(batch):
        images[i, :, :, :img.shape[-1]] = img
        images[i, :, :, img.shape[-1]:] = img[:, :, -2:].median()
    texts = [t for _, t in batch]
    targets = [encode(t) for t in texts]
    return {"images": images,
            "input_lengths": torch.tensor([img.shape[-1] // STRIDE for img, _ in batch]),
            "targets": torch.tensor([i for t in targets for i in t], dtype=torch.long),
            "target_lengths": torch.tensor([len(t) for t in targets]),
            "texts": texts}


# ---------------------------------------------------------------------------------------- sample sheet
def sample_sheet(split: str, severity: int, n: int, seed: int, path: Path, tough: bool = False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = SyntheticLines(split, n, seed=seed, severity=severity, tough=tough)
    fig, axes = plt.subplots(n, 1, figsize=(9, 0.62 * n))
    for ax, i in zip(np.atleast_1d(axes), range(n)):
        image, text = data.item(i)
        ax.imshow(image, cmap="gray", vmin=0, vmax=255, aspect="equal")
        ax.set_title(text, fontsize=7, loc="left", family="monospace", pad=2)
        ax.axis("off")
    fig.suptitle(f"Synthetic lines ({split} split, severity {severity}{', tough' if tough else ''})", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--split", default="train", choices=list(SPLIT_OFFSET))
    parser.add_argument("--severity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tough", action="store_true", help="add the tougher print/photo effects")
    args = parser.parse_args()
    name = f"synthetic_{args.split}_severity{args.severity}{'_tough' if args.tough else ''}.png"
    out = project_path(f"reports/figures/ocr_model/{name}")
    print("saved", sample_sheet(args.split, args.severity, args.samples, args.seed, out, args.tough))


if __name__ == "__main__":
    main()
