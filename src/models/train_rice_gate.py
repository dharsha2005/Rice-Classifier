"""Train a binary rice-vs-non-rice gate using real images in datasets/rice_gate/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.rice_gate import (
    MODEL_PATH,
    METADATA_PATH,
    NON_RICE_DIR,
    RICE_DIR,
    DEFAULT_ACCEPTANCE_THRESHOLD,
    extract_gate_features,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _collect_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def load_dataset() -> tuple[list[Path], np.ndarray]:
    rice_images = _collect_images(RICE_DIR)
    non_rice_images = _collect_images(NON_RICE_DIR)
    if not rice_images:
        raise FileNotFoundError(f"No rice images found in {RICE_DIR}")
    if not non_rice_images:
        raise FileNotFoundError(
            f"No non-rice images found in {NON_RICE_DIR}. "
            "Add real non-rice photos (hands, desk, objects, empty scenes) before training."
        )
    paths = rice_images + non_rice_images
    labels = np.array([1] * len(rice_images) + [0] * len(non_rice_images), dtype=np.int64)
    return paths, labels


def extract_feature_matrix(paths: list[Path]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for index, path in enumerate(paths, start=1):
        rows.append(extract_gate_features(path).reshape(-1))
        if index % 25 == 0 or index == len(paths):
            print(f"  extracted features for {index}/{len(paths)} images")
    return np.vstack(rows).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train rice-vs-non-rice binary gate model.")
    parser.add_argument(
        "--acceptance-threshold",
        type=float,
        default=DEFAULT_ACCEPTANCE_THRESHOLD,
        help="Probability threshold for accepting rice at inference time.",
    )
    parser.add_argument("--test-size", type=float, default=0.15, help="Held-out test fraction.")
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation fraction from the training pool.")
    args = parser.parse_args()

    paths, labels = load_dataset()
    print(f"Loaded {int(labels.sum())} rice and {int((labels == 0).sum())} non-rice images.")

    features = extract_feature_matrix(paths)
    x_train, x_test, y_train, y_test, paths_train, paths_test = train_test_split(
        features,
        labels,
        paths,
        test_size=args.test_size,
        random_state=42,
        stratify=labels,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=args.val_size,
        random_state=42,
        stratify=y_train,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    model.fit(x_train_scaled, y_train)

    def evaluate(split_name: str, x_split: np.ndarray, y_split: np.ndarray) -> dict[str, float | str]:
        probabilities = model.predict_proba(x_split)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metrics: dict[str, float | str] = {
            "accuracy": float(accuracy_score(y_split, predictions)),
            "precision": float(precision_score(y_split, predictions, zero_division=0)),
            "recall": float(recall_score(y_split, predictions, zero_division=0)),
            "f1": float(f1_score(y_split, predictions, zero_division=0)),
        }
        if len(np.unique(y_split)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_split, probabilities))
        print(f"\n{split_name} metrics:")
        print(json.dumps(metrics, indent=2))
        print(classification_report(y_split, predictions, target_names=["non_rice", "rice"]))
        print("Confusion matrix:\n", confusion_matrix(y_split, predictions))
        return metrics

    val_metrics = evaluate("Validation", x_val_scaled, y_val)
    test_metrics = evaluate("Test", x_test_scaled, y_test)

    bundle = {"scaler": scaler, "classifier": model}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    metadata = {
        "acceptance_threshold": float(args.acceptance_threshold),
        "feature_dim": int(features.shape[1]),
        "rice_dir": str(RICE_DIR),
        "non_rice_dir": str(NON_RICE_DIR),
        "train_images": int(len(y_train)),
        "val_images": int(len(y_val)),
        "test_images": int(len(y_test)),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved gate model -> {MODEL_PATH}")
    print(f"Saved metadata   -> {METADATA_PATH}")


if __name__ == "__main__":
    main()
