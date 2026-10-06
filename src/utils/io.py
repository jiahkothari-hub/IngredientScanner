"""Reading and writing JSON Lines (one JSON object per line) - the format used for all NER data.

JSONL is human-readable, streams well and loads directly with Hugging Face:
    from datasets import Dataset
    ds = Dataset.from_json("data/splits/silver/train.jsonl.gz")   # .gz is read directly
"""
import gzip
import json
from pathlib import Path


def _open(path, mode):
    """Plain or gzip-compressed text file, decided by the file name (.gz)."""
    if str(path).endswith(".gz"):
        # mtime=0 makes the compressed bytes identical on every run (reproducible files in git)
        if "w" in mode:
            return gzip.GzipFile(filename="", mode="wb", fileobj=open(path, "wb"), mtime=0)
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def write_jsonl(records, path) -> int:
    """Write one JSON object per line. A path ending in .gz is gzip-compressed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with _open(path, "w") as f:
        for record in records:
            line = json.dumps(record, ensure_ascii=False) + "\n"
            f.write(line.encode("utf-8") if str(path).endswith(".gz") else line)
            n += 1
    return n


def read_jsonl(path) -> list:
    with _open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]
