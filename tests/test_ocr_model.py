"""Tests for the deep-learning text-line recogniser: synthetic data, the three models, CTC decoding, metrics.
Run:  pytest tests/test_ocr_model.py -q"""
import random

import numpy as np
import pytest
import torch

from src.ocr_model.charset import BLANK, CHAR2ID, NUM_CLASSES, clean_text, decode_ids, encode
from src.ocr_model.evaluate import error_rates
from src.ocr_model.models import MODELS, build, count_parameters, greedy_decode
from src.ocr_model.render_lines import (HEIGHT, MAX_WIDTH, STRIDE, SyntheticLines, collate, fit_width, font_paths,
                                        render, sample_line, split_texts)
from src.utils.config import project_path

needs_data = pytest.mark.skipif(not project_path("data/splits/silver/train.jsonl.gz").exists(),
                                reason="silver splits not built")


# ------------------------------------------------------------------ characters
def test_clean_text_maps_into_the_character_set():
    assert clean_text("Crème  fraîche – 12%") == "Creme fraiche - 12%"
    assert clean_text("sugar • salt") == "sugar , salt"
    assert all(ch in CHAR2ID for ch in clean_text("Ünïcode ½ ©text"))


def test_encode_decode_round_trip_and_blank_is_zero():
    text = "Acidity Regulator (INS 330), 12%"
    ids = encode(text)
    assert BLANK == 0 and 0 not in ids
    assert decode_ids(ids) == text
    assert NUM_CLASSES == len(CHAR2ID) + 1


# ------------------------------------------------------------------ synthetic lines
@needs_data
def test_synthetic_lines_are_reproducible_and_model_ready():
    data = SyntheticLines("train", 5, seed=3)
    a, b = data.item(2), data.item(2)
    assert a[1] == b[1] and np.array_equal(a[0], b[0])
    image, text = data[0]
    assert image.shape[0] == 1 and image.shape[1] == HEIGHT
    assert image.shape[2] % STRIDE == 0 and image.shape[2] <= MAX_WIDTH
    assert 0 <= image.min() and image.max() <= 1
    assert text and all(ch in CHAR2ID for ch in text)


@needs_data
def test_splits_do_not_share_fonts():
    assert not set(font_paths("train")) & set(font_paths("test"))


@needs_data
def test_sample_line_is_a_run_of_whole_words():
    rng = random.Random(0)
    texts = split_texts("validation")
    for _ in range(20):
        line = sample_line(texts, rng)
        assert 1 <= len(line) <= 48
        assert line == line.strip()


def test_fit_width_leaves_room_for_ctc():
    text = "coffee"                       # 6 characters, 2 repeats -> needs 4 * (6 + 2 + 2) = 40 columns
    img = fit_width(np.zeros((HEIGHT, 10), np.uint8), text)
    assert img.shape[1] >= STRIDE * 10 and img.shape[1] % STRIDE == 0


def test_render_works_at_every_severity():
    font = font_paths("train")[0]
    for severity in (0, 1, 2):
        img = render("Sugar, Palm Oil", font, random.Random(severity), severity)
        assert img.dtype == np.uint8 and img.shape[0] == HEIGHT


# ------------------------------------------------------------------ models
@pytest.mark.parametrize("name", list(MODELS))
def test_models_give_one_step_per_four_columns(name):
    model = build(name).eval()
    with torch.no_grad():
        out = model(torch.rand(2, 1, HEIGHT, 120))
    assert out.shape == (120 // STRIDE, 2, NUM_CLASSES)
    assert torch.allclose(out.exp().sum(-1), torch.ones(30, 2), atol=1e-4)      # log-probabilities
    assert count_parameters(model) > 100_000


@pytest.mark.parametrize("name", list(MODELS))
def test_one_training_step_reduces_ctc_loss_on_a_fixed_batch(name):
    torch.manual_seed(0)
    images = torch.rand(4, 1, HEIGHT, 160)
    targets = [encode("salt"), encode("sugar"), encode("oil"), encode("E330")]
    batch = dict(targets=torch.tensor([i for t in targets for i in t]),
                 input_lengths=torch.full((4,), 160 // STRIDE), target_lengths=torch.tensor([len(t) for t in targets]))
    model, ctc = build(name), torch.nn.CTCLoss(zero_infinity=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(15):
        loss = ctc(model(images), batch["targets"], batch["input_lengths"], batch["target_lengths"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_greedy_decode_merges_repeats_and_drops_blanks():
    a, b = CHAR2ID["o"], CHAR2ID["f"]
    steps = [BLANK, b, b, BLANK, a, a, BLANK, a, BLANK]          # "f", "o", blank, "o" -> "foo"
    log_probs = torch.full((len(steps), 1, NUM_CLASSES), -10.0)
    for t, c in enumerate(steps):
        log_probs[t, 0, c] = 0.0
    assert greedy_decode(log_probs) == ["foo"]
    assert greedy_decode(log_probs, lengths=[4]) == ["f"]


@needs_data
def test_collate_pads_and_packs_ctc_targets():
    data = SyntheticLines("validation", 3, seed=0)
    batch = collate([data[i] for i in range(3)])
    widths = [data[i][0].shape[-1] for i in range(3)]
    assert batch["images"].shape == (3, 1, HEIGHT, max(widths))
    assert batch["input_lengths"].tolist() == [w // STRIDE for w in widths]
    assert batch["target_lengths"].sum() == len(batch["targets"])


# ------------------------------------------------------------------ metrics
def test_error_rates():
    r = error_rates(["sugar salt", "palm oii"], ["sugar, salt", "palm oil"])
    assert r["cer"] == pytest.approx(2 / 19)                     # one missing comma, one wrong letter
    assert r["wer"] == pytest.approx(2 / 4)                       # "sugar" != "sugar,", "oii" != "oil"
    assert r["line_accuracy"] == 0
    assert error_rates(["E330"], ["E330"])["cer"] == 0


# ------------------------------------------------------------------ real-photo labels
from src.ocr_model.real_photos import best_span, label_crop, strip_header, widen_to_words  # noqa: E402

REF = ("Sugar, Palm Oil, Hazelnuts (13%), Skimmed Milk Powder (8.7%), Fat-Reduced Cocoa (7.4%), "
       "Emulsifier: Lecithins (Soya), Vanillin")


def test_best_span_finds_the_place_despite_ocr_errors():
    s, e, sim = best_span("Ska1mmed Mi1k Powder (8.7", REF)
    assert REF[s:e] == "Skimmed Milk Powder (8.7" and sim > 0.8
    assert best_span("BRAND NAME 400g", REF)[2] < 0.5


def test_label_is_the_union_of_both_readings_widened_to_words():
    # one reader drops the end of the line: the label still comes from the other, longer reading
    assert label_crop(["Milk Powder (8.7%), Fat-Red", "Milk Powder (8.7%), Fat-Reduced Coc"], REF) == \
        "Milk Powder (8.7%), Fat-Reduced Cocoa"
    assert widen_to_words(REF, REF.index("zelnuts"), REF.index("zelnuts") + 3) == \
        (REF.index("Hazelnuts"), REF.index("Hazelnuts") + len("Hazelnuts"))


def test_short_or_unrelated_readings_get_no_label():
    assert label_crop(["Per", "Per"], REF) is None
    assert label_crop(["Best before end of month", "Best before end"], REF) is None


def test_strip_header():
    assert strip_header("INGREDIENTS: Sugar, salt") == "Sugar, salt"
    assert strip_header("Ingredient- Sugar") == "Sugar"
    assert strip_header("Sugar, ingredients") == "Sugar, ingredients"


# ------------------------------------------------------------------ fine-tuning on real crops
def test_photo_halves_split_by_photo_and_cover_everything():
    import pandas as pd
    from src.ocr_model.finetune import photo_halves
    rows = pd.DataFrame({"product_id": ["a", "a", "b", "c", "c", "d", "e"]})
    first, second = photo_halves(rows)
    assert not first & second and first | second == {"a", "b", "c", "d", "e"}
    assert photo_halves(rows) == (first, second)                   # same split every time


def test_real_crop_augmentation_keeps_height_and_type():
    from src.ocr_model.finetune import augment, real_item
    img = (np.random.default_rng(0).random((HEIGHT, 120)) * 255).astype(np.uint8)
    out = augment(img, random.Random(1))
    assert out.dtype == np.uint8 and out.shape[0] == HEIGHT
    tensor, label = real_item(out, "sugar, salt")
    assert tensor.shape[1] == HEIGHT and tensor.shape[2] % STRIDE == 0 and label == "sugar, salt"


def test_tough_lines_render():
    data = SyntheticLines("validation", 4, seed=0, tough=True)
    for i in range(4):
        image, text = data.item(i)
        assert image.shape[0] == HEIGHT and text


def test_misaligned_flags_only_line_ends():
    from src.ocr_model.ppocr_finetune import misaligned
    assert misaligned("Sea Salt, Organic Black Pepper,", "ts: Sea Salt, Organic Black Pepper,")   # heading fragment
    assert misaligned("How to cook Cooking", "How to cook Coo")                                 # cut-off end
    assert not misaligned("Maltodextrin, Mint Flavor", "Maltodetrin, Mint Rlavor")             # misread inside
    assert misaligned("salt", "")


def test_combination_prefers_the_more_ingredient_like_reading():
    import math
    from src.ocr_model.combine import combine_line, combine_words, plausibility
    assert plausibility("Sugar, Palm Oil, Cocoa Powder") > plausibility("Sugor, Palm Oll, Cocoa Powder")
    assert combine_line("Sugor, Palm Oll", "Sugar, Palm Oil", 0.0) == "Sugar, Palm Oil"
    assert combine_line("Sugor, Palm Oll", "Sugar, Palm Oil", math.inf) == "Sugor, Palm Oll"   # never replace
    assert combine_words("Acid), MIIk Proteins.", "Acid), Milk Proteins.", 0.0) == "Acid), Milk Proteins."


def test_beam_search_matches_greedy_without_lm_and_fixes_spelling_with_it():
    from src.ocr_model.combine import resources
    from src.ocr_model.improve_crnn import beam_decode
    lm, _ = resources()
    seq = [0, CHAR2ID["s"], CHAR2ID["u"], 0, CHAR2ID["g"], CHAR2ID["o"], CHAR2ID["r"], 0]
    lp = np.full((len(seq), NUM_CLASSES), -8.0, np.float32)
    for t, c in enumerate(seq):
        lp[t, c] = -0.05
    lp[5, CHAR2ID["a"]] = -0.9                        # 'a' is the runner-up where the model read 'o'
    assert beam_decode(lp, lm, 0.0, 0.0) == greedy_decode(torch.from_numpy(lp)[:, None])[0] == "sugor"
    assert beam_decode(lp, lm, 1.0, 0.5) == "sugar"


def test_every_optimiser_in_the_study_can_train_the_crnn():
    from src.ocr_model.train import OPTIMISERS, make_optimiser
    for name in OPTIMISERS:
        model = build("crnn")
        opt = make_optimiser(name, model.parameters(), 1e-3)
        loss = model(torch.rand(2, 1, HEIGHT, 64)).mean()
        loss.backward()
        opt.step()


@pytest.mark.skipif(not project_path("models/ocr-crnn/model.pt").exists(), reason="app CRNN not trained")
def test_app_can_read_a_photo_with_our_crnn():
    from src.ocr.ocr_pipeline import read_image
    r = read_image(project_path("data/images/packets/8901063162518.jpg"), recognizer="crnn")
    assert r["engine"].startswith("Our CRNN")
    assert "WHEAT FLOUR" in r["text"].upper()
