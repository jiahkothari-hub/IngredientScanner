"""Fine-tune DistilBERT / BERT for ingredient NER (token classification), with optional CRF layer
and optional OCR-noise data augmentation.

Usage (from the project root):
    python -m src.ner.transformer_ner --model distilbert-base-uncased --run distilbert
    python -m src.ner.transformer_ner --model distilbert-base-uncased --augment --run distilbert_aug
    python -m src.ner.transformer_ner --model bert-base-cased --run bert --max-train 8000
    python -m src.ner.transformer_ner --model distilbert-base-uncased --crf --run distilbert_crf

How it works, step by step:
  1. Our data has one BIO tag per WORD. The Transformer splits words into sub-words
     ("glucose" -> "glucose", "maltodextrin" -> "malt", "##ode", "##xt", "##rin").
     We give each word's FIRST sub-word the word's label and mark the other sub-words with -100,
     which PyTorch ignores in the loss (the standard Hugging Face recipe).
  2. The pretrained encoder turns every sub-word into a 768-number vector that depends on the whole
     sentence (self-attention). A linear layer maps each vector to 21 scores, one per BIO tag.
  3. Training minimises cross-entropy between those scores and the silver labels (AdamW, linear
     learning-rate warm-up/decay, gradient clipping).
  4. --crf: instead of choosing every word's tag independently, a CRF layer learns transition scores
     between tags (I-SUGAR may follow B-SUGAR, never O) and decodes the best whole sequence (Viterbi).
  5. --augment: every epoch, half of the training sentences get OCR-style character noise
     (l<->1, O<->0, ...) or a case change, so the model sees text like Person 3's OCR output.

Runs are saved in models/runs/<run>/ (Hugging Face format + training log).
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.evaluation.metrics import evaluate_tag_sequences
from src.evaluation.noise_robustness import add_ocr_noise
from src.labeling.bio import ID2LABEL, LABEL2ID, bio_to_entities
from src.ner.data import load_split, tags_of
from src.ner.pretrained import ensure_pretrained
from src.preprocessing.pipeline import process_ingredient_text
from src.utils.config import project_path

RUNS_DIR = project_path("models/runs")
IGNORE = -100
TRAIN_MAX_LENGTH = 256     # sub-words; covers ~99% of lists, keeps CPU training affordable
PREDICT_MAX_LENGTH = 512   # BERT's maximum


# ---------------------------------------------------------------------------- encoding
def encode(tokenizer, words: list, tags: list = None, max_length: int = TRAIN_MAX_LENGTH) -> dict:
    """Tokenise one pre-split sentence and align word labels to the first sub-word of each word."""
    enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=max_length)
    word_ids = enc.word_ids()
    first = [i for i, w in enumerate(word_ids) if w is not None and (i == 0 or word_ids[i - 1] != w)]
    item = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "first_subword": first}
    if tags is not None:
        labels = [IGNORE] * len(word_ids)
        for i in first:
            labels[i] = LABEL2ID[tags[word_ids[i]]]
        item["labels"] = labels
    return item


def collate(items: list, pad_id: int) -> dict:
    """Pad a batch to its longest member (dynamic padding)."""
    width = max(len(it["input_ids"]) for it in items)
    batch = {
        "input_ids": torch.tensor([it["input_ids"] + [pad_id] * (width - len(it["input_ids"])) for it in items]),
        "attention_mask": torch.tensor([it["attention_mask"] + [0] * (width - len(it["attention_mask"]))
                                        for it in items]),
    }
    if "labels" in items[0]:
        batch["labels"] = torch.tensor([it["labels"] + [IGNORE] * (width - len(it["labels"])) for it in items])
    return batch


def length_bucketed_batches(items: list, batch_size: int, rng: random.Random) -> list:
    """Group sentences of similar length (less padding = faster on CPU), then shuffle the batches."""
    order = sorted(range(len(items)), key=lambda i: len(items[i]["input_ids"]) + rng.random())
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    rng.shuffle(batches)
    return batches


# ---------------------------------------------------------------------------- augmentation
def augment_record(record: dict, rng: random.Random) -> list:
    """Return the words of a noisy/case-changed copy of the sentence (labels stay the same)."""
    text = record["text"]
    choice = rng.random()
    if choice < 0.6:
        text = add_ocr_noise(text, rng.choice([0.02, 0.05, 0.10]), rng)
    elif choice < 0.8:
        text = text.upper()
    else:
        text = text.lower()
    return [text[s:e] for s, e in record["token_offsets"]]


# ---------------------------------------------------------------------------- CRF variant
class TransformerCRF(nn.Module):
    """Token-classification Transformer + CRF over the WORD sequence (first sub-word of each word)."""

    def __init__(self, encoder):
        super().__init__()
        from torchcrf import CRF
        self.encoder = encoder
        self.crf = CRF(len(LABEL2ID), batch_first=True)

    def word_emissions(self, input_ids, attention_mask, labels):
        logits = self.encoder(input_ids=input_ids, attention_mask=attention_mask).logits
        mask = labels != IGNORE                                   # first sub-words only
        n_words = mask.sum(1)
        width = int(n_words.max())
        emissions = logits.new_zeros(logits.size(0), width, logits.size(2))
        word_labels = labels.new_zeros(labels.size(0), width)
        word_mask = torch.zeros(labels.size(0), width, dtype=torch.bool)
        for b in range(logits.size(0)):
            k = int(n_words[b])
            emissions[b, :k] = logits[b][mask[b]]
            word_labels[b, :k] = labels[b][mask[b]]
            word_mask[b, :k] = True
        return emissions, word_labels, word_mask

    def forward(self, input_ids, attention_mask, labels):
        emissions, word_labels, word_mask = self.word_emissions(input_ids, attention_mask, labels)
        return -self.crf(emissions, word_labels, mask=word_mask, reduction="mean")

    def decode(self, input_ids, attention_mask, labels):
        emissions, _, word_mask = self.word_emissions(input_ids, attention_mask, labels)
        return self.crf.decode(emissions, mask=word_mask)


# ---------------------------------------------------------------------------- prediction
class TransformerTagger:
    """Load a fine-tuned run and tag sentences. Same interface as the other systems."""

    def __init__(self, model_dir, device="cpu"):
        self.model_dir = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        model = AutoModelForTokenClassification.from_pretrained(self.model_dir, dtype=torch.float32)
        self.crf = None
        if (self.model_dir / "crf.pt").exists():
            self.model = TransformerCRF(model)
            self.model.crf.load_state_dict(torch.load(self.model_dir / "crf.pt"))
        else:
            self.model = model
        self.model.eval().to(device)
        self.name = self.model_dir.name

    @torch.no_grad()
    def predict_records(self, records: list, batch_size: int = 32) -> list:
        return self.predict_token_lists([r["tokens"] for r in records], batch_size)

    @torch.no_grad()
    def predict_token_lists(self, token_lists: list, batch_size: int = 32) -> list:
        results = [None] * len(token_lists)
        order = sorted(range(len(token_lists)), key=lambda i: len(token_lists[i]))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            items = []
            for i in idx:
                item = encode(self.tokenizer, token_lists[i] or ["."], [ "O"] * max(1, len(token_lists[i])),
                              max_length=PREDICT_MAX_LENGTH)
                items.append(item)
            batch = collate(items, self.tokenizer.pad_token_id)
            if isinstance(self.model, TransformerCRF):
                paths = self.model.decode(batch["input_ids"], batch["attention_mask"], batch["labels"])
            else:
                logits = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
                paths = [logits[b, items[b]["first_subword"]].argmax(-1).tolist() for b in range(len(idx))]
            for b, i in enumerate(idx):
                tags = [ID2LABEL[t] for t in paths[b]]
                n = len(token_lists[i])
                results[i] = (tags + ["O"] * n)[:n]          # words beyond 512 sub-words -> O
        return [fix_bio(t) for t in results]

    def predict_tags(self, tokens: list) -> list:
        return self.predict_token_lists([tokens])[0]

    def predict(self, text: str) -> dict:
        """Raw text -> normalised text, tokens and entities (like DictionaryNER.predict)."""
        processed = process_ingredient_text(text)
        tags = self.predict_tags(processed["tokens"]) if processed["tokens"] else []
        processed["entities"] = bio_to_entities(processed["token_offsets"], tags, processed["text"])
        processed["tags"] = tags
        return processed


def fix_bio(tags: list) -> list:
    """Repair impossible sequences from the softmax head (I-X after O or after another type -> B-X)."""
    fixed, previous = [], "O"
    for tag in tags:
        if tag.startswith("I-") and previous[2:] != tag[2:]:
            tag = "B-" + tag[2:]
        fixed.append(tag)
        previous = tag
    return fixed


# ---------------------------------------------------------------------------- training
def evaluate_run(tagger_like, records) -> dict:
    predictions = tagger_like.predict_records(records)
    return evaluate_tag_sequences([tags_of(r) for r in records], predictions)


def train(args):
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    torch.set_num_threads(args.threads)
    base = ensure_pretrained(args.model)
    out_dir = RUNS_DIR / args.run
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForTokenClassification.from_pretrained(
        base, num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID)
    net = TransformerCRF(model) if args.crf else model

    train_records = load_split("silver", "train", limit=args.max_train, seed=args.seed)
    val_records = load_split("silver", "validation", limit=args.max_val, seed=args.seed)
    clean_items = [encode(tokenizer, r["tokens"], tags_of(r)) for r in train_records]
    print(f"{args.run}: {len(train_records):,} train / {len(val_records):,} validation sentences", flush=True)

    encoder_params = [p for n, p in net.named_parameters() if not n.startswith("crf.")]
    groups = [{"params": encoder_params, "lr": args.lr}]
    if args.crf:
        groups.append({"params": net.crf.parameters(), "lr": 1e-3})   # transitions learn faster
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(clean_items) / args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps_per_epoch * args.epochs),
                                                steps_per_epoch * args.epochs)
    log, best_f1, start = {"args": vars(args), "epochs": []}, -1.0, time.time()

    for epoch in range(1, args.epochs + 1):
        if args.augment:   # fresh noise every epoch for half of the sentences
            items = [encode(tokenizer, augment_record(r, rng), tags_of(r)) if rng.random() < 0.5 else clean
                     for r, clean in zip(train_records, clean_items)]
        else:
            items = clean_items
        net.train()
        total_loss = 0.0
        for step, batch_idx in enumerate(length_bucketed_batches(items, args.batch_size, rng), start=1):
            batch = collate([items[i] for i in batch_idx], tokenizer.pad_token_id)
            if args.crf:
                loss = net(**batch)
            else:
                loss = net(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{steps_per_epoch} loss {total_loss / step:.4f} "
                      f"({(time.time() - start) / 60:.1f} min)", flush=True)

        # validation: save the model only if it is the best so far
        save_run(net, tokenizer, out_dir, args.crf)
        val = evaluate_run(TransformerTagger(out_dir), val_records)
        entry = {"epoch": epoch, "train_loss": round(total_loss / steps_per_epoch, 4),
                 "val_f1": val["micro"]["f1"], "minutes": round((time.time() - start) / 60, 1)}
        log["epochs"].append(entry)
        print(f"  epoch {epoch}: {entry}", flush=True)
        if val["micro"]["f1"] > best_f1:
            best_f1 = val["micro"]["f1"]
            save_run(net, tokenizer, out_dir / "best", args.crf)
    log["best_val_f1"] = best_f1
    (out_dir / "best" / "training_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"{args.run}: best validation F1 {best_f1:.4f}; saved {out_dir / 'best'}")


def save_run(net, tokenizer, folder: Path, crf: bool):
    folder.mkdir(parents=True, exist_ok=True)
    encoder = net.encoder if crf else net
    encoder.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    if crf:
        torch.save(net.crf.state_dict(), folder / "crf.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--run", required=True, help="name of the output folder in models/runs/")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-train", type=int, default=None, help="subsample training sentences (CPU time)")
    parser.add_argument("--max-val", type=int, default=800)
    parser.add_argument("--crf", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
