from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_HANDCRAFTED_PATH = PROJECT_ROOT / "results" / "features" / "train_handcrafted_features.csv"
TRAIN_DEEP_PATH = PROJECT_ROOT / "results" / "features" / "deep" / "train_efficientnet_features.csv"
METADATA_COLUMNS = ["image_path", "split", "class_id", "class_name"]


def _read_feature_columns(path: Path, prefix: str | None = None) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing training feature file: {path}")
    columns = list(pd.read_csv(path, nrows=0).columns)
    names = [column for column in columns if column not in METADATA_COLUMNS]
    if prefix is not None:
        names = [column for column in names if column.startswith(prefix)]
    return names


def build_feature_schema() -> list[str]:
    handcrafted = _read_feature_columns(TRAIN_HANDCRAFTED_PATH)
    deep = _read_feature_columns(TRAIN_DEEP_PATH, prefix="deep_feature_")

    if len(handcrafted) != 62:
        raise ValueError(f"Expected 62 handcrafted features, found {len(handcrafted)}")
    if len(deep) != 1280:
        raise ValueError(f"Expected 1280 deep features, found {len(deep)}")

    schema = handcrafted + deep
    if len(schema) != 1342:
        raise ValueError(f"Expected 1342 fused features, found {len(schema)}")
    return schema


FEATURE_SCHEMA = build_feature_schema()
HANDCRAFTED_FEATURE_SCHEMA = FEATURE_SCHEMA[:62]
DEEP_FEATURE_SCHEMA = FEATURE_SCHEMA[62:]
FEATURE_GROUP_COUNTS = {
    "shape": 14,
    "glcm": 12,
    "colour": 36,
    "deep": 1280,
    "total": 1342,
}

__all__ = [
    "FEATURE_SCHEMA",
    "HANDCRAFTED_FEATURE_SCHEMA",
    "DEEP_FEATURE_SCHEMA",
    "FEATURE_GROUP_COUNTS",
    "build_feature_schema",
]
