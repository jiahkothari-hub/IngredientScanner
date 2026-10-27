"""Train one text-line recogniser (ann / cnn / crnn) with the CTC loss on synthetic label lines.

    python -m src.ocr_model.train --model crnn                 # full run (~12k steps, a few hours on CPU)
    python -m src.ocr_model.train --model ann --steps 300 --run smoke_ann   # quick check

All three models get exactly the same data, loss, optimiser and schedule, so the comparison isolates the
architecture:
  * data       a fresh synthetic image for every training example (never repeated), validation lines from
               validation-split products, test lines from test-split products in fonts never used for training
  * loss       CTC (connectionist temporal classification): learns from the image + its text alone, without
               knowing where each character is
  * optimiser  AdamW by default (--optimizer: sgd, momentum, nesterov, rmsprop, adam, adamw; see optimizer_study.py),
               one-cycle learning-rate schedule, gradient clipping at 5 (keeps LSTM training stable)
  * selection  the checkpoint with the lowest validation CER is kept; training stops early when validation
               CER has not improved for `patience` evaluations
Outputs: models/runs/ocr/<run>/ (best.pt, history.csv) and reports/ocr_model/<run>.json (all results).
"""
import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
from torch import nn

from src.ocr_model.evaluate import evaluate
from src.ocr_model.models import MODELS, build, count_parameters
from src.ocr_model.render_lines import SyntheticLines, collate
from src.utils.config import project_path


def _worker_init(_):
    torch.set_num_threads(1)          # data workers render images; leave the cores to the model


OPTIMISERS = ("sgd", "momentum", "nesterov", "rmsprop", "adam", "adamw")


def make_optimiser(name: str, params, lr: float, weight_decay: float = 1e-4):
    """sgd: plain mini-batch SGD | momentum: SGD + momentum 0.9 | nesterov: SGD + Nesterov momentum 0.9 |
    rmsprop | adam (no weight decay) | adamw (decoupled weight decay 1e-4, the default)."""
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=weight_decay)
    if name == "momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "nesterov":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, alpha=0.99, momentum=0.0, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(name)


def train(args) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    run = args.run or args.model
    out_dir = project_path(f"models/runs/ocr/{run}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    model = build(args.model).to(device)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    optimiser = make_optimiser(args.optimizer, model.parameters(), args.lr)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=args.lr, total_steps=args.steps, pct_start=0.1,
                                                   cycle_momentum=args.cycle_momentum and args.optimizer != "sgd")

    train_data = SyntheticLines("train", args.steps * args.batch_size, seed=args.seed, severity=1, tough=args.tough)
    loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                                         num_workers=args.workers, worker_init_fn=_worker_init,
                                         persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None)
    validation = SyntheticLines("validation", args.val_size, seed=1, severity=1, tough=args.tough)

    print(f"{args.model}: {count_parameters(model):,} parameters, {args.steps} steps x {args.batch_size} lines "
          f"on {device}", flush=True)
    history, best, bad_evals, started, running = [], None, 0, time.time(), []
    for step, batch in enumerate(loader, start=1):
        model.train()
        log_probs = model(batch["images"].to(device, non_blocking=True))
        loss = ctc(log_probs, batch["targets"].to(device), batch["input_lengths"].to(device),
                   batch["target_lengths"].to(device))
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
        schedule.step()
        running.append(loss.item())

        if step % args.eval_every == 0 or step == args.steps:
            val = evaluate(model, validation, ctc_loss=ctc)
            row = {"step": step, "train_loss": round(float(np.mean(running)), 4), "val_loss": round(val["loss"], 4),
                   "val_cer": round(val["cer"], 4), "val_wer": round(val["wer"], 4),
                   "lr": schedule.get_last_lr()[0], "minutes": round((time.time() - started) / 60, 1)}
            history.append(row)
            running = []
            print("  ".join(f"{k} {v}" for k, v in row.items()), flush=True)
            if best is None or val["cer"] < best["val_cer"]:
                best, bad_evals = row, 0
                torch.save({"model": args.model, "state_dict": model.state_dict(), "step": step, "args": vars(args)},
                           out_dir / "best.pt")
            else:
                bad_evals += 1
                if bad_evals >= args.patience:
                    print(f"early stop: validation CER has not improved for {args.patience} evaluations")
                    break

    with open(out_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    # final test on held-out products AND held-out fonts, at three photo-damage levels
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device)["state_dict"])
    test = {f"severity_{s}": evaluate(model, SyntheticLines("test", args.test_size, seed=2, severity=s))
            for s in (0, 1, 2)}
    result = {"run": run, "model": args.model, "parameters": count_parameters(model), "best": best,
              "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
              "train_minutes": round((time.time() - started) / 60, 1), "steps_done": history[-1]["step"],
              "settings": {k: v for k, v in vars(args).items()}, "test": test, "history": history}
    report = project_path(f"reports/ocr_model/{run}.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2))
    print("test CER  " + "  ".join(f"{k}: {v['cer']:.3f}" for k, v in test.items()))
    print("saved", report)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--run", default=None, help="run name (default: the model name)")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=min(3, max(0, (os.cpu_count() or 2) // 4)))
    parser.add_argument("--threads", type=int, default=0, help="torch threads for the model (0 = default)")
    parser.add_argument("--device", default="auto", help="auto (GPU if available), cpu or cuda")
    parser.add_argument("--tough", action="store_true", help="train/validate on the tougher synthetic lines")
    parser.add_argument("--optimizer", default="adamw", choices=OPTIMISERS)
    parser.add_argument("--no-cycle-momentum", dest="cycle_momentum", action="store_false",
                        help="keep momentum / beta1 fixed instead of cycling it with the one-cycle schedule")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
