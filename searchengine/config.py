"""Project-wide settings, read from environment variables with sane defaults.

Kept deliberately small: every script takes CLI flags, and these are only the
defaults those flags fall back to.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _path(env: str, default: str) -> Path:
    p = Path(os.environ.get(env, default))
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass(frozen=True)
class Settings:
    es_url: str
    data_dir: Path
    index_dir: Path
    runs_dir: Path


def get_settings() -> Settings:
    return Settings(
        es_url=os.environ.get("ES_URL", "http://127.0.0.1:9200"),
        data_dir=_path("DATA_DIR", "data/processed"),
        index_dir=_path("INDEX_DIR", "indexes"),
        runs_dir=_path("RUNS_DIR", "runs"),
    )
