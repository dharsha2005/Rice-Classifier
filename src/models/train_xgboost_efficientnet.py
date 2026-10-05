"""Train and evaluate XGBoost using EfficientNet-B0 features only."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = PROJECT_ROOT / "results" / "features" / "deep"
OUTPUT_DIR = PROJECT_ROOT / "results" / "models" / "efficientnet_xgboost"
SVM_REPORT = PROJECT_ROOT / "results" / "models" / "svm" / "svm_baseline_report.txt"
COMPARISON_DIR = PROJECT_ROOT / "results" / "models"
CLASS_NAMES = {0: "0_NOR", 1: "1_F&S", 2: "2_SD", 3: "3_MY", 4: "4_AP", 5: "5_BN", 6: "6_UN", 7: "7_IM"}
METADATA_COLUMNS = ["image_path", "split", "class_id", "class_name"]
SPLITS = ["train", "val", "test"]
RANDOM_STATE = 42


def load_deep_features(feature_dir: Path = FEATURE_DIR) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    frames: Dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        path = feature_dir / f"{split}_efficientnet_features.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing EfficientNet feature file: {path}")
        frame = pd.read_csv(path)
        feature_columns = [column for column in frame.columns if column.startswith("deep_feature_")]
        if not feature_columns:
            raise ValueError(f"No deep_feature_* columns found in {path}")
        frames[split] = frame
    feature_columns = [column for column in frames["train"].columns if column.startswith("deep_feature_")]
    if any([column for column in frame.columns if column.startswith("deep_feature_")] != feature_columns for frame in frames.values()):
        raise ValueError("Deep feature columns are inconsistent across splits")
    return frames, feature_columns


def validate_data(frames: Dict[str, pd.DataFrame], feature_columns: Sequence[str]) -> Dict[str, Dict[int, int]]:
    distributions: Dict[str, Dict[int, int]] = {}
    expected_ids = set(CLASS_NAMES)
    for split, frame in frames.items():
        if frame[METADATA_COLUMNS].isna().any().any():
            raise ValueError(f"Missing metadata values in {split} split")
        values = frame[list(feature_columns)].to_numpy(dtype=np.float64)
        if np.isnan(values).any() or np.isinf(values).any():
            raise ValueError(f"NaN or infinite deep features found in {split} split")
        labels = set(frame["class_id"].astype(int).unique())
        if not labels.issubset(expected_ids):
            raise ValueError(f"Unexpected class IDs in {split}: {sorted(labels - expected_ids)}")
        if not frame["image_path"].is_unique:
            raise ValueError(f"Duplicate image paths found in {split} split")
        if not (frame["class_name"] == frame["class_id"].map(CLASS_NAMES)).all():
            raise ValueError(f"Class names do not match class IDs in {split} split")
        if not (frame["split"] == split).all():
            raise ValueError(f"Split metadata mismatch in {split} feature file")
        distributions[split] = frame["class_id"].astype(int).value_counts().reindex(sorted(CLASS_NAMES), fill_value=0).to_dict()
    return distributions


def make_sample_weights(labels: np.ndarray, balanced: bool) -> np.ndarray | None:
    if not balanced:
        return None
    classes, counts = np.unique(labels, return_counts=True)
    weights = {int(cls): len(labels) / (len(classes) * count) for cls, count in zip(classes, counts)}
    return np.asarray([weights[int(label)] for label in labels], dtype=np.float32)


def build_classifier(params: Dict[str, Any]) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASS_NAMES),
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=4,
        random_state=RANDOM_STATE,
        verbosity=0,
        **params,
    )


def tune_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[Dict[str, Any], bool, pd.DataFrame]:
    search_space = {
        "n_estimators": [50, 100],
        "max_depth": [4],
        "learning_rate": [0.1],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "min_child_weight": [1],
        "balanced_sample_weight": [False, True],
    }
    records: List[Dict[str, Any]] = []
    total = 4
    configuration = 0
    for n_estimators in search_space["n_estimators"]:
        for max_depth in search_space["max_depth"]:
            for learning_rate in search_space["learning_rate"]:
                for balanced in search_space["balanced_sample_weight"]:
                    configuration += 1
                    params = {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "learning_rate": learning_rate,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "min_child_weight": 1,
                    }
                    model = build_classifier(params)
                    weights = make_sample_weights(y_train, balanced)
                    started = time.perf_counter()
                    model.fit(X_train, y_train, sample_weight=weights)
                    predictions = model.predict(X_val)
                    elapsed = time.perf_counter() - started
                    record = {
                        **params,
                        "balanced_sample_weight": balanced,
                        "validation_accuracy": accuracy_score(y_val, predictions),
                        "validation_macro_f1": f1_score(y_val, predictions, average="macro", zero_division=0),
                        "validation_weighted_f1": f1_score(y_val, predictions, average="weighted", zero_division=0),
                        "validation_macro_precision": precision_score(y_val, predictions, average="macro", zero_division=0),
                        "validation_macro_recall": recall_score(y_val, predictions, average="macro", zero_division=0),
                        "fit_seconds": elapsed,
                    }
                    records.append(record)
                    print(f"[{configuration:02d}/{total}] depth={max_depth} trees={n_estimators} lr={learning_rate} balanced={balanced} | val accuracy={record['validation_accuracy']:.4f} macro_f1={record['validation_macro_f1']:.4f}")
    results = pd.DataFrame(records).sort_values(["validation_macro_f1", "validation_accuracy"], ascending=False).reset_index(drop=True)
    best = results.iloc[0].to_dict()
    best_params = {key: best[key] for key in search_space if key != "balanced_sample_weight"}
    best_params = {key: (int(value) if key in {"n_estimators", "max_depth", "min_child_weight"} else float(value)) for key, value in best_params.items()}
    return best_params, bool(best["balanced_sample_weight"]), results


def evaluate(model: XGBClassifier, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    started = time.perf_counter()
    predictions = model.predict(X)
    elapsed = time.perf_counter() - started
    labels = sorted(CLASS_NAMES)
    matrix = confusion_matrix(y, predictions, labels=labels)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix.astype(float), row_totals, out=np.zeros_like(matrix, dtype=float), where=row_totals != 0)
    per_class = pd.DataFrame({
        "class_id": labels,
        "class_name": [CLASS_NAMES[label] for label in labels],
        "precision": precision_score(y, predictions, labels=labels, average=None, zero_division=0),
        "recall": recall_score(y, predictions, labels=labels, average=None, zero_division=0),
        "f1_score": f1_score(y, predictions, labels=labels, average=None, zero_division=0),
        "support": np.bincount(y, minlength=len(labels)),
    })
    return {
        "accuracy": accuracy_score(y, predictions),
        "macro_precision": precision_score(y, predictions, average="macro", zero_division=0),
        "macro_recall": recall_score(y, predictions, average="macro", zero_division=0),
        "macro_f1": f1_score(y, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, predictions, average="weighted", zero_division=0),
        "confusion_matrix": matrix,
        "normalized_confusion_matrix": normalized,
        "per_class": per_class,
        "classification_report": classification_report(y, predictions, labels=labels, target_names=[CLASS_NAMES[label] for label in labels], digits=4, zero_division=0),
        "inference_seconds": elapsed,
        "milliseconds_per_image": elapsed / len(y) * 1000.0,
    }


def create_plots(results: Dict[str, Any], tuning: pd.DataFrame, model: XGBClassifier, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [CLASS_NAMES[index] for index in sorted(CLASS_NAMES)]
    plt.style.use("dark_background")
    for filename, matrix, title, fmt, values in [
        ("confusion_matrix.png", results["confusion_matrix"], "XGBoost + EfficientNet Features: Test Confusion Matrix", "d", results["confusion_matrix"]),
        ("normalized_confusion_matrix.png", results["normalized_confusion_matrix"] * 100.0, "XGBoost + EfficientNet Features: Normalized Test Confusion Matrix", ".1f", results["normalized_confusion_matrix"] * 100.0),
    ]:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt=fmt, cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax, cbar_kws={"label": "Count" if fmt == "d" else "Recall (%)"})
        ax.set_title(title)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)
    per_class = results["per_class"]
    plot_data = per_class.melt(id_vars=["class_name"], value_vars=["precision", "recall", "f1_score"], var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=plot_data, x="class_name", y="score", hue="metric", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Class Metrics: XGBoost + EfficientNet Deep Features")
    ax.set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(output_dir / "per_class_metrics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(["Accuracy", "Macro F1", "Weighted F1"], [results["accuracy"], results["macro_f1"], results["weighted_f1"]], color=["#38bdf8", "#10b981", "#f59e0b"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Test Performance")
    axes[1].barh(np.arange(len(model.feature_importances_)), model.feature_importances_, color="#a855f7")
    axes[1].set_title("Top EfficientNet Deep-Feature Importances")
    axes[1].set_xlabel("Importance")
    top = np.argsort(model.feature_importances_)[-15:]
    axes[1].set_yticks(np.arange(len(model.feature_importances_))[top])
    axes[1].set_yticklabels([f"deep_feature_{index + 1:03d}" for index in top], fontsize=8)
    fig.suptitle("XGBoost + EfficientNet-B0 Deep Features")
    fig.tight_layout()
    fig.savefig(output_dir / "performance_summary.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(1, len(tuning) + 1), tuning["validation_macro_f1"].to_numpy(), marker=".")
    ax.set_title("Validation Macro F1 Across Tested Configurations")
    ax.set_xlabel("Configuration rank")
    ax.set_ylabel("Validation macro F1")
    fig.tight_layout()
    fig.savefig(output_dir / "validation_tuning.png", dpi=180)
    plt.close(fig)


def read_svm_metrics(path: Path) -> Dict[str, float]:
    text = path.read_text(encoding="utf-8")
    patterns = {"accuracy": r"Test Accuracy\s+:\s+([0-9.]+)%", "macro_f1": r"Macro F1-Score\s+:\s+([0-9.]+)", "weighted_f1": r"Weighted F1-Score\s+:\s+([0-9.]+)"}
    values: Dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"Could not read SVM {key} from {path}")
        value = float(match.group(1))
        values[key] = value / 100.0 if key == "accuracy" else value
    return values


def save_comparison(xgb_results: Dict[str, Any], output_dir: Path) -> Tuple[Path, Path, Dict[str, float]]:
    svm = read_svm_metrics(SVM_REPORT)
    xgb = {"accuracy": xgb_results["accuracy"], "macro_f1": xgb_results["macro_f1"], "weighted_f1": xgb_results["weighted_f1"]}
    comparison = pd.DataFrame([
        {"Model": "SVM", "Feature Type": "62 handcrafted features", "Accuracy": svm["accuracy"], "Macro F1": svm["macro_f1"], "Weighted F1": svm["weighted_f1"]},
        {"Model": "XGBoost", "Feature Type": "EfficientNet-B0 deep features", "Accuracy": xgb["accuracy"], "Macro F1": xgb["macro_f1"], "Weighted F1": xgb["weighted_f1"]},
    ])
    csv_path = output_dir / "model_comparison_baseline.csv"
    comparison.to_csv(csv_path, index=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison.set_index("Model")[["Accuracy", "Macro F1", "Weighted F1"]].plot(kind="bar", ax=ax, color=["#38bdf8", "#10b981", "#f59e0b"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Baseline Comparison: Handcrafted SVM vs EfficientNet XGBoost")
    ax.legend(loc="lower right")
    fig.tight_layout()
    image_path = output_dir / "model_comparison_baseline.png"
    fig.savefig(image_path, dpi=180)
    plt.close(fig)
    return csv_path, image_path, svm


def write_report(path: Path, frames: Dict[str, pd.DataFrame], distributions: Dict[str, Dict[int, int]], feature_dimension: int, search_space: Dict[str, Any], best_params: Dict[str, Any], balanced: bool, validation: Dict[str, float], test: Dict[str, Any], svm: Dict[str, float], elapsed: float) -> None:
    test_class = test["per_class"]
    deep_better_macro = test["macro_f1"] > svm["macro_f1"]
    deep_better_accuracy = test["accuracy"] > svm["accuracy"]
    lines = [
        "EFFICIENTNET-B0 DEEP FEATURES + XGBOOST REPORT", "",
        "Research role: controlled classifier experiment using EfficientNet-B0 deep features only.",
        "No handcrafted features, SVM, feature fusion, or final-model claim is included.", "",
        f"Dataset sizes: train={len(frames['train'])}, validation={len(frames['val'])}, test={len(frames['test'])}",
        f"Deep feature dimension: {feature_dimension}", f"Number of classes: {len(CLASS_NAMES)}", "",
        "Class distribution:",
    ]
    for split in SPLITS:
        lines.append(f"  {split}: " + ", ".join(f"{CLASS_NAMES[class_id]}={distributions[split][class_id]}" for class_id in sorted(CLASS_NAMES)))
    lines.extend([
        "", "XGBoost configuration:", "  objective=multi:softprob, tree_method=hist, random_state=42, n_jobs=4",
        f"  balanced inverse-frequency sample weights selected: {balanced}",
        f"Hyperparameter search space: {json.dumps(search_space, sort_keys=True)}",
        f"Best hyperparameters: {json.dumps(best_params, sort_keys=True)}",
        "", "Validation performance:", f"  Accuracy: {validation['validation_accuracy']:.4f}", f"  Macro F1: {validation['validation_macro_f1']:.4f}", f"  Weighted F1: {validation['validation_weighted_f1']:.4f}",
        "", "Final test performance:", f"  Accuracy: {test['accuracy']:.4f}", f"  Macro Precision: {test['macro_precision']:.4f}", f"  Macro Recall: {test['macro_recall']:.4f}", f"  Macro F1: {test['macro_f1']:.4f}", f"  Weighted F1: {test['weighted_f1']:.4f}", f"  Inference time: {test['inference_seconds']:.4f} seconds ({test['milliseconds_per_image']:.4f} ms/image)",
        "", "Per-class test metrics:", "  " + test_class.to_string(index=False), "", "Classification report:", test["classification_report"],
        "Confusion matrix rows are true classes and columns are predicted classes:",
    ])
    for class_id, row in zip(sorted(CLASS_NAMES), test["confusion_matrix"]):
        lines.append(f"  True {CLASS_NAMES[class_id]:<8}: " + " ".join(str(int(value)) for value in row))
    lines.extend([
        "", "Comparison with SVM baseline:", f"  SVM test accuracy={svm['accuracy']:.4f}, macro F1={svm['macro_f1']:.4f}, weighted F1={svm['weighted_f1']:.4f}", f"  XGBoost deep-feature test accuracy={test['accuracy']:.4f}, macro F1={test['macro_f1']:.4f}, weighted F1={test['weighted_f1']:.4f}", f"  Deep features outperform handcrafted SVM on accuracy: {deep_better_accuracy}; on macro F1: {deep_better_macro}.",
        "  The lower-support defect classes remain the most sensitive to class imbalance; macro F1 is therefore emphasized.",
        "  A hybrid model remains necessary to investigate because this experiment evaluates deep features alone and does not establish whether complementary handcrafted information improves results.",
        f"Total feature-loading/training experiment time: {elapsed:.3f} seconds", "",
        "Feature importance note: importance values refer to EfficientNet deep-feature dimensions, not original image features.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost on EfficientNet-B0 features only")
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    started = time.perf_counter()
    frames, feature_columns = load_deep_features(args.feature_dir)
    distributions = validate_data(frames, feature_columns)
    X_train = frames["train"][feature_columns].to_numpy(dtype=np.float32)
    y_train = frames["train"]["class_id"].to_numpy(dtype=np.int32)
    X_val = frames["val"][feature_columns].to_numpy(dtype=np.float32)
    y_val = frames["val"]["class_id"].to_numpy(dtype=np.int32)
    X_test = frames["test"][feature_columns].to_numpy(dtype=np.float32)
    y_test = frames["test"]["class_id"].to_numpy(dtype=np.int32)
    best_params, balanced, tuning = tune_models(X_train, y_train, X_val, y_val)
    best_row = tuning.iloc[0]
    validation = {column: float(best_row[column]) for column in ["validation_accuracy", "validation_macro_f1", "validation_weighted_f1"]}
    final_X = np.vstack([X_train, X_val])
    final_y = np.concatenate([y_train, y_val])
    final_model = build_classifier(best_params)
    final_model.fit(final_X, final_y, sample_weight=make_sample_weights(final_y, balanced))
    test = evaluate(final_model, X_test, y_test)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, args.output_dir / "xgboost_efficientnet.joblib")
    (args.output_dir / "best_parameters.json").write_text(json.dumps({**best_params, "balanced_sample_weight": balanced}, indent=2) + "\n", encoding="utf-8")
    create_plots(test, tuning, final_model, args.output_dir)
    comparison_csv, comparison_png, svm = save_comparison(test, COMPARISON_DIR)
    search_space = {"n_estimators": [50, 100], "max_depth": [4], "learning_rate": [0.1], "subsample": [0.8], "colsample_bytree": [0.8], "min_child_weight": [1], "balanced_sample_weight": [False, True]}
    report = args.output_dir / "efficientnet_xgboost_report.txt"
    write_report(report, frames, distributions, len(feature_columns), search_space, best_params, balanced, validation, test, svm, time.perf_counter() - started)
    print("\nXGBOOST EFFICIENTNET EXPERIMENT COMPLETED")
    print(f"Deep feature dimension: {len(feature_columns)}")
    print(f"Best parameters: {json.dumps({**best_params, 'balanced_sample_weight': balanced}, sort_keys=True)}")
    print(f"Validation accuracy: {validation['validation_accuracy']:.4f}; validation macro F1: {validation['validation_macro_f1']:.4f}")
    print(f"Test accuracy: {test['accuracy']:.4f}; test macro F1: {test['macro_f1']:.4f}; test weighted F1: {test['weighted_f1']:.4f}")
    print(test["per_class"].to_string(index=False))
    print(f"Inference time: {test['inference_seconds']:.4f}s")
    print(f"Comparison CSV: {comparison_csv}")
    print(f"Comparison image: {comparison_png}")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
