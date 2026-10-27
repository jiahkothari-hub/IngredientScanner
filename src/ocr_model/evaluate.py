"""Character / word error rates of a recogniser on a set of line images.

CER = character edits (insert, delete, substitute) needed to turn the prediction into the reference,
      divided by the number of reference characters (summed over all lines, so long lines count more).
WER = the same at word level. Line accuracy = share of lines read exactly right.
"""
import time

import torch
from rapidfuzz.distance import Levenshtein

from src.ocr_model.models import greedy_decode
from src.ocr_model.render_lines import collate


def error_rates(predictions: list, references: list) -> dict:
    char_edits = sum(Levenshtein.distance(p, r) for p, r in zip(predictions, references))
    word_edits = sum(Levenshtein.distance(p.split(), r.split()) for p, r in zip(predictions, references))
    chars = sum(len(r) for r in references)
    words = sum(len(r.split()) for r in references)
    return {"cer": char_edits / max(chars, 1), "wer": word_edits / max(words, 1),
            "line_accuracy": sum(p == r for p, r in zip(predictions, references)) / max(len(references), 1),
            "lines": len(references)}


@torch.no_grad()
def evaluate(model, dataset, batch_size: int = 64, ctc_loss=None) -> dict:
    """Error rates (and mean CTC loss if ctc_loss is given) plus a few example readings."""
    model.eval()
    device = next(model.parameters()).device
    predictions, references, losses, seconds = [], [], [], 0.0
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate)
    for batch in loader:
        start = time.perf_counter()
        log_probs = model(batch["images"].to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds += time.perf_counter() - start
        log_probs = log_probs.float().cpu()
        if ctc_loss is not None:
            losses.append(ctc_loss(log_probs, batch["targets"], batch["input_lengths"], batch["target_lengths"]).item())
        predictions += greedy_decode(log_probs, batch["input_lengths"])
        references += batch["texts"]
    result = error_rates(predictions, references)
    result["ms_per_line"] = 1000 * seconds / max(len(references), 1)
    if losses:
        result["loss"] = sum(losses) / len(losses)
    result["examples"] = [{"reference": r, "prediction": p} for p, r in zip(predictions[:12], references[:12])]
    return result
