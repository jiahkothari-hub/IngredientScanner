"""Download the official Open Food Facts CSV export.

Usage (from the project root):
    python -m src.data.download_dataset            # download if not already present
    python -m src.data.download_dataset --force    # re-download

Why the full export and not the search API?
  * The API is rate-limited (about 10 search requests/minute) and is meant for apps,
    not bulk downloads. OFF asks people who need many products to use the exports.
  * The export is a single gzip file (~1.3 GB compressed). We download it once and
    afterwards only ever read ~10 of its ~200 columns.

Reproducibility:
  Open Food Facts is updated every day, so two downloads on different days differ.
  We therefore write a manifest (URL, date, size, SHA-256) next to the file. The
  filtered project dataset produced later is committed to git, so the whole team
  works on the same frozen snapshot.
"""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone

import requests

from src.utils.config import load_config, project_path

CHUNK_SIZE = 1024 * 1024  # 1 MB


def sha256_of_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(url: str, destination) -> int:
    """Stream `url` to `destination` (via a .part file) and return the number of bytes."""
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        written = 0
        last_report = time.time()
        with open(tmp_path, "wb") as f:
            for block in response.iter_content(chunk_size=CHUNK_SIZE):
                f.write(block)
                written += len(block)
                if time.time() - last_report > 10:  # progress every 10 s
                    pct = f"{100 * written / total:.1f}%" if total else "?"
                    print(f"  {written / 1e6:,.0f} MB downloaded ({pct})", flush=True)
                    last_report = time.time()
    if total and written != total:
        raise IOError(f"Incomplete download: got {written} of {total} bytes")
    tmp_path.replace(destination)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = parser.parse_args()

    config = load_config()
    destination = project_path(config["source"]["raw_file"])
    manifest_path = project_path(config["source"]["manifest_file"])
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not args.force:
        print(f"{destination} already exists - skipping download (use --force to re-download).")
        return

    errors = []
    for url in config["source"]["urls"]:
        print(f"Downloading {url}")
        try:
            size = download_file(url, destination)
            break
        except (requests.RequestException, IOError) as exc:
            print(f"  failed: {exc}")
            errors.append(f"{url}: {exc}")
    else:
        raise SystemExit("All download URLs failed:\n" + "\n".join(errors))

    manifest = {
        "url": url,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "size_bytes": size,
        "sha256": sha256_of_file(destination),
        "license": config["source"]["license"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {destination} ({size / 1e6:,.0f} MB). Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
