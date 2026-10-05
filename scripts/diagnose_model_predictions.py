"""Predict one sample per class folder and print raw probabilities for pipeline verification."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.feature_schema import FEATURE_GROUP_COUNTS
from src.inference.predict import (
    CLASS_LABELS,
    create_hybrid_features,
    decode_proba_column,
    load_scaler,
    load_xgboost_model,
    predict_rice,
)


def main() -> int:
    image_root = PROJECT_ROOT / "rice" / "test"
    if not image_root.is_dir():
        print(f"Missing test image root: {image_root}", file=sys.stderr)
        return 1

    model = load_xgboost_model()
    scaler = load_scaler()
    print("model.classes_", list(getattr(model, "classes_", [])))
    print("scaler.n_features_in_", getattr(scaler, "n_features_in_", None))
    print("feature groups", FEATURE_GROUP_COUNTS)

    predictions: list[str] = []
    for class_name in CLASS_LABELS:
        folder = image_root / class_name
        images = sorted(folder.glob("*.png"))
        if not images:
            print(f"ERROR: no PNG samples in {folder}", file=sys.stderr)
            return 1
        image_path = images[0]
        hybrid = create_hybrid_features(image_path)
        if hybrid.shape != (1342,):
            print(f"ERROR: hybrid shape {hybrid.shape} for {image_path}", file=sys.stderr)
            return 1
        scaled = scaler.transform(hybrid.reshape(1, -1))
        probabilities = model.predict_proba(scaled)[0]
        if not np.isclose(probabilities.sum(), 1.0, atol=1e-5):
            print(f"ERROR: probabilities do not sum to 1 for {image_path}: {probabilities.sum()}", file=sys.stderr)
            return 1
        column = int(np.argmax(probabilities))
        decoded = decode_proba_column(column, model)
        pipeline = predict_rice(image_path)
        predictions.append(pipeline["predicted_class"])
        print(f"\n[{class_name}] sample={image_path.name}")
        print("  raw probabilities:", np.round(probabilities, 4))
        print("  argmax column:", column, "decoded:", decoded, "confidence:", round(float(probabilities[column]), 4))
        print("  predict_rice:", pipeline["predicted_class"], pipeline["confidence"])

    unique = set(predictions)
    print(f"\nUnique predicted classes across folders: {sorted(unique)}")
    if len(unique) == 1:
        print("FAIL: all eight folders collapsed to a single prediction.", file=sys.stderr)
        return 2
    print("PASS: model outputs vary across class folders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
