"""Small helpers for locating the project root and loading the YAML config.

Every script resolves paths relative to the project root, so the pipeline works
the same on Windows, macOS and Linux regardless of where it is run from.
"""
from pathlib import Path

import yaml

# src/utils/config.py -> parents[2] is the project root (the repository root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data_config.yaml"


def load_config(path=DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(relative: str) -> Path:
    """Turn a config path such as 'data/raw/x.csv' into an absolute path."""
    return PROJECT_ROOT / relative
