"""Download pretrained Transformer weights (DistilBERT, BERT) for fine-tuning.

Usage:  python -m src.ner.pretrained distilbert-base-uncased bert-base-cased

The files come from Hugging Face's original public S3 bucket (models.huggingface.co), the location
the `transformers` library itself used before the Hub. They are the same official checkpoints
(Sanh et al. 2019 for DistilBERT, Devlin et al. 2019 for BERT). We download them with plain HTTPS
so the project also works where huggingface.co is not reachable.

Result:  models/pretrained/<name>/{config.json, vocab.txt, pytorch_model.bin}
which `from_pretrained("models/pretrained/<name>")` loads like any Hub model.
"""
import json
import sys
from pathlib import Path

import requests

from src.utils.config import project_path

BASE = "https://s3.amazonaws.com/models.huggingface.co/bert"
MODELS = {
    # name: (weights/config prefix, vocab file, lower-case?)
    "distilbert-base-uncased": ("distilbert-base-uncased", "bert-base-uncased-vocab.txt", True),
    "bert-base-cased": ("bert-base-cased", "bert-base-cased-vocab.txt", False),
    "bert-base-uncased": ("bert-base-uncased", "bert-base-uncased-vocab.txt", True),
}
PRETRAINED_DIR = project_path("models/pretrained")


def _download(url: str, path: Path):
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        tmp = path.with_suffix(path.suffix + ".part")
        with open(tmp, "wb") as f:
            for block in response.iter_content(chunk_size=1 << 20):
                f.write(block)
        tmp.replace(path)


def ensure_pretrained(name: str) -> Path:
    """Download `name` if it is not there yet and return its folder."""
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {list(MODELS)}")
    prefix, vocab, lower = MODELS[name]
    folder = PRETRAINED_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    files = {"config.json": f"{BASE}/{prefix}-config.json",
             "vocab.txt": f"{BASE}/{vocab}",
             "pytorch_model.bin": f"{BASE}/{prefix}-pytorch_model.bin"}
    for file_name, url in files.items():
        if not (folder / file_name).exists():
            print(f"  downloading {name}/{file_name}", flush=True)
            _download(url, folder / file_name)
    # tokenizer settings, so AutoTokenizer.from_pretrained(folder) works
    tokenizer_config = {"do_lower_case": lower, "model_max_length": 512}
    (folder / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
    return folder


if __name__ == "__main__":
    for model_name in sys.argv[1:] or ["distilbert-base-uncased"]:
        print(ensure_pretrained(model_name))
