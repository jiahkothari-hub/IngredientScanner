"""Training vs validation loss curve for the DistilBERT NER recipe (for the report and slides).

    python -m src.ner.loss_curve

Repeats the exact recipe of the app's model (run distilbert_aug in transformer_ner.py: distilbert-base-uncased,
8,000 training lists, 2 epochs, batch 16, AdamW 5e-5, weight decay 0.01, 10% linear warm-up then linear decay,
gradient clipping 1.0, OCR-noise augmentation on half the lists each epoch, seed 42), and every EVAL_EVERY steps
measures the cross-entropy loss on the 800 validation lists and on the 1,000 clean test lists. The model itself
is not saved (the app keeps models/ingredient-ner-distilbert). Uses the GPU when there is one.
Writes reports/distilbert_loss_curve.json.
"""
import json
import math
import random
import time

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.labeling.bio import ID2LABEL, LABEL2ID
from src.ner.data import evaluation_sets, load_split, tags_of
from src.ner.pretrained import ensure_pretrained
from src.ner.transformer_ner import augment_record, collate, encode, length_bucketed_batches
from src.utils.config import project_path

EVAL_EVERY = 50


@torch.no_grad()
def mean_loss(net, items, pad_id, device, batch_size: int = 64) -> float:
    """Average cross-entropy per labelled word (sub-word continuations ignored)."""
    net.eval()
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    total, words = 0.0, 0
    for i in range(0, len(items), batch_size):
        b = collate(items[i:i + batch_size], pad_id)
        logits = net(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device)).logits
        labels = b["labels"].to(device)
        total += ce(logits.view(-1, logits.shape[-1]), labels.view(-1)).item()
        words += (labels != -100).sum().item()
    net.train()
    return total / words


def main(seed: int = 42, epochs: int = 2, batch_size: int = 16, lr: float = 5e-5):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = ensure_pretrained("distilbert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(base)
    net = AutoModelForTokenClassification.from_pretrained(base, num_labels=len(LABEL2ID), id2label=ID2LABEL,
                                                          label2id=LABEL2ID).to(device)
    train_records = load_split("silver", "train", limit=8000, seed=seed)
    val_items = [encode(tokenizer, r["tokens"], tags_of(r)) for r in load_split("silver", "validation", limit=800, seed=seed)]
    test_items = [encode(tokenizer, r["tokens"], tags_of(r), max_length=512)
                  for r in evaluation_sets(limit=1000)["silver_test"]]
    clean_items = [encode(tokenizer, r["tokens"], tags_of(r)) for r in train_records]

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(clean_items) / batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps_per_epoch * epochs), steps_per_epoch * epochs)
    pad = tokenizer.pad_token_id
    history, step, window, start = [], 0, [], time.time()
    history.append({"step": 0, "epoch": 0, "train_loss": None, "val_loss": round(mean_loss(net, val_items, pad, device), 4),
                    "test_loss": round(mean_loss(net, test_items, pad, device), 4)})
    net.train()
    for epoch in range(1, epochs + 1):
        items = [encode(tokenizer, augment_record(r, rng), tags_of(r)) if rng.random() < 0.5 else clean
                 for r, clean in zip(train_records, clean_items)]
        for batch_idx in length_bucketed_batches(items, batch_size, rng):
            b = collate([items[i] for i in batch_idx], pad)
            loss = net(**{k: v.to(device) for k, v in b.items()}).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            window.append(loss.item())
            if step % EVAL_EVERY == 0 or step == steps_per_epoch * epochs:
                row = {"step": step, "epoch": round(step / steps_per_epoch, 2),
                       "train_loss": round(sum(window) / len(window), 4),
                       "val_loss": round(mean_loss(net, val_items, pad, device), 4),
                       "test_loss": round(mean_loss(net, test_items, pad, device), 4),
                       "minutes": round((time.time() - start) / 60, 1)}
                history.append(row)
                window = []
                print(row, flush=True)
    out = project_path("reports/distilbert_loss_curve.json")
    out.write_text(json.dumps({"recipe": "distilbert_aug", "eval_every": EVAL_EVERY, "device": str(device),
                               "history": history}, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
