"""Train and evaluate the MAIN HYBRID FEATURE-FUSION MODEL (1342 features -> XGBoost).

Project:
    Rice Quality and Defect Assessment using Hybrid Machine Learning and Deep Feature Fusion

Description:
    Combines 62 handcrafted features (14 Shape, 12 GLCM Texture, 36 Colour) with
    1280 EfficientNet-B0 deep features into a 1342-dimensional hybrid feature vector.
    Trains and evaluates XGBoost multiclass classifier with leakage-safe scaling,
    controlled hyperparameter tuning, class-imbalance experimentation, feature domain
    importance breakdown, ablation study, and complete reporting.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
from sklearn.preprocessing import RobustScaler, StandardScaler
from xgboost import XGBClassifier

# Constants & Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
HANDCRAFTED_FEATURE_DIR = PROJECT_ROOT / "results" / "features"
DEEP_FEATURE_DIR = PROJECT_ROOT / "results" / "features" / "deep"
HYBRID_FEATURE_DIR = PROJECT_ROOT / "results" / "features" / "hybrid"
OUTPUT_DIR = PROJECT_ROOT / "results" / "models" / "hybrid_xgboost"

CLASS_NAMES = {
    0: "0_NOR",
    1: "1_F&S",
    2: "2_SD",
    3: "3_MY",
    4: "4_AP",
    5: "5_BN",
    6: "6_UN",
    7: "7_IM",
}

METADATA_COLUMNS = ["image_path", "split", "class_id", "class_name"]
SPLITS = ["train", "val", "test"]
EXPECTED_COUNTS = {"train": 24767, "val": 3095, "test": 3100}
RANDOM_STATE = 42

# Baseline Experiment Benchmarks
BASELINE_SVM = {"accuracy": 0.9116, "macro_f1": 0.8494, "weighted_f1": 0.9161}
BASELINE_EFFICIENTNET = {"accuracy": 0.9065, "macro_f1": 0.8399, "weighted_f1": 0.9099}


def validate_source_split(df: pd.DataFrame, split: str, source_name: str) -> None:
    """Fail fast on source-data integrity problems; never repair feature data."""
    expected = EXPECTED_COUNTS[split]
    if len(df) != expected:
        raise ValueError(f"{source_name} {split}: expected {expected} rows, found {len(df)}")
    if df["image_path"].isna().any() or not df["image_path"].is_unique:
        raise ValueError(f"{source_name} {split}: image_path contains missing or duplicate values")
    if not (df["split"] == split).all():
        raise ValueError(f"{source_name} {split}: split metadata does not match its source file")
    if not (df["class_name"] == df["class_id"].map(CLASS_NAMES)).all():
        raise ValueError(f"{source_name} {split}: class_id/class_name metadata mismatch")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    values = df[numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{source_name} {split}: NaN or infinite numeric values found")


def load_and_align_features(
    handcrafted_dir: Path = HANDCRAFTED_FEATURE_DIR,
    deep_dir: Path = DEEP_FEATURE_DIR,
    hybrid_dir: Path = HYBRID_FEATURE_DIR,
) -> Tuple[Dict[str, pd.DataFrame], List[str], Dict[str, List[str]]]:
    """
    Load handcrafted and deep feature CSVs, strictly verify metadata, and merge on explicit keys.
    Always merge the source tables explicitly on image_path and verify all metadata.
    """
    frames: Dict[str, pd.DataFrame] = {}

    # Source tables are always joined explicitly; cached hybrid files are not trusted.
    hybrid_paths = [hybrid_dir / f"{split}_hybrid_features.csv" for split in SPLITS]
    if False and all(p.exists() for p in hybrid_paths):
        print("Pre-merged hybrid feature CSV files found in results/features/hybrid/. Loading directly...")
        for split, path in zip(SPLITS, hybrid_paths):
            df = pd.read_csv(path)
            expected = EXPECTED_COUNTS[split]
            if len(df) != expected:
                raise ValueError(f"Hybrid feature file {path} row count mismatch: expected {expected}, got {len(df)}")
            frames[split] = df
    else:
        print("Performing explicit metadata key join between handcrafted and deep feature CSVs...")
        for split in SPLITS:
            h_path = handcrafted_dir / f"{split}_handcrafted_features.csv"
            d_path = deep_dir / f"{split}_efficientnet_features.csv"

            if not h_path.exists():
                raise FileNotFoundError(f"Missing handcrafted feature file: {h_path}")
            if not d_path.exists():
                raise FileNotFoundError(f"Missing deep feature file: {d_path}")

            df_h = pd.read_csv(h_path)
            df_d = pd.read_csv(d_path)

            for meta_col in METADATA_COLUMNS:
                if meta_col not in df_h.columns or meta_col not in df_d.columns:
                    raise ValueError(f"Missing metadata column '{meta_col}' in {split} files")

            validate_source_split(df_h, split, "handcrafted features")
            validate_source_split(df_d, split, "EfficientNet features")

            if set(df_h["image_path"]) != set(df_d["image_path"]):
                raise ValueError(f"image_path mismatch in {split} split")
            for key in ["split", "class_id", "class_name"]:
                h_meta = df_h.set_index("image_path")[key].sort_index()
                d_meta = df_d.set_index("image_path")[key].sort_index()
                if not h_meta.equals(d_meta):
                    raise ValueError(f"{key} mismatch in {split} split")

            merged = pd.merge(
                df_h,
                df_d.drop(columns=["split", "class_id", "class_name"]),
                on="image_path",
                how="inner",
                validate="one_to_one",
            )

            expected = EXPECTED_COUNTS[split]
            if len(merged) != expected:
                raise ValueError(
                    f"Feature alignment row count mismatch for '{split}': expected {expected}, got {len(merged)}"
                )

            frames[split] = merged

    # The predefined split membership is part of the experimental protocol.
    all_paths: set[str] = set()
    for split, frame in frames.items():
        if not frame["image_path"].is_unique or len(frame) != EXPECTED_COUNTS[split]:
            raise ValueError(f"Merged {split} split failed uniqueness or sample-count validation")
        overlap = all_paths.intersection(frame["image_path"])
        if overlap:
            raise ValueError(f"Split membership changed: {len(overlap)} image_path values occur in multiple splits")
        all_paths.update(frame["image_path"])

    # Extract & organize feature column names from train split
    sample_df = frames["train"]
    feature_cols = [c for c in sample_df.columns if c not in METADATA_COLUMNS]

    shape_cols = [c for c in feature_cols if c.startswith("shape_")]
    glcm_cols = [c for c in feature_cols if c.startswith("glcm_")]
    color_cols = [c for c in feature_cols if c.startswith("color_")]
    deep_cols = [c for c in feature_cols if c.startswith("deep_feature_")]

    if len(feature_cols) != 1342:
        raise ValueError(f"Expected 1342 total hybrid features, found {len(feature_cols)}")
    if len(shape_cols) != 14:
        raise ValueError(f"Expected 14 shape features, found {len(shape_cols)}")
    if len(glcm_cols) != 12:
        raise ValueError(f"Expected 12 GLCM features, found {len(glcm_cols)}")
    if len(color_cols) != 36:
        raise ValueError(f"Expected 36 colour features, found {len(color_cols)}")
    if len(deep_cols) != 1280:
        raise ValueError(f"Expected 1280 deep features, found {len(deep_cols)}")

    feature_groups = {
        "Shape": shape_cols,
        "GLCM Texture": glcm_cols,
        "Colour": color_cols,
        "EfficientNet Deep": deep_cols,
    }

    print("Data loading & explicit feature alignment verified successfully:")
    for split in SPLITS:
        print(f"  - {split.capitalize()}: {len(frames[split])} images, {len(feature_cols)} hybrid features")

    return frames, feature_cols, feature_groups


def save_hybrid_features(frames: Dict[str, pd.DataFrame], output_dir: Path = HYBRID_FEATURE_DIR) -> None:
    """Save the merged hybrid feature CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        path = output_dir / f"{split}_hybrid_features.csv"
        frames[split].to_csv(path, index=False)
        print(f"Saved hybrid features split: {path} ({path.stat().st_size / (1024*1024):.2f} MB)")


def get_scaler(scaler_name: str | None) -> Any:
    """Instantiate a feature scaler instance."""
    if scaler_name == "StandardScaler":
        return StandardScaler()
    elif scaler_name == "RobustScaler":
        return RobustScaler()
    elif scaler_name is None or scaler_name == "None":
        return None
    else:
        raise ValueError(f"Unknown scaler_name: {scaler_name}")


def make_sample_weights(labels: np.ndarray, balanced: bool) -> np.ndarray | None:
    """Compute balanced inverse class frequency sample weights."""
    if not balanced:
        return None
    classes, counts = np.unique(labels, return_counts=True)
    weights = {int(cls): len(labels) / (len(classes) * count) for cls, count in zip(classes, counts)}
    return np.asarray([weights[int(label)] for label in labels], dtype=np.float32)


def build_classifier(params: Dict[str, Any]) -> XGBClassifier:
    """Construct XGBClassifier instance with clean standard settings."""
    return XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASS_NAMES),
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0,
        **params,
    )


def tune_hybrid_model(
    frames: Dict[str, pd.DataFrame],
    feature_cols: Sequence[str],
) -> Tuple[Dict[str, Any], bool, str | None, pd.DataFrame]:
    """
    Perform lightweight validation-based search over scalers, hyperparams, and class weighting.
    Selection primary metric: Validation Macro F1.
    """
    y_train = frames["train"]["class_id"].to_numpy(dtype=int)
    y_val = frames["val"]["class_id"].to_numpy(dtype=int)

    X_train_raw = frames["train"][list(feature_cols)].to_numpy(dtype=np.float64)
    X_val_raw = frames["val"][list(feature_cols)].to_numpy(dtype=np.float64)

    search_space = {
        # Two protocol-required experiments in the established best parameter region:
        # A: unweighted; B: inverse-frequency sample weighted.
        "n_estimators": [100],
        "max_depth": [4],
        "learning_rate": [0.1],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "min_child_weight": [1],
        "scaler_name": ["StandardScaler"],
        "balanced_sample_weight": [False, True],
    }

    records: List[Dict[str, Any]] = []
    total_configs = (
        len(search_space["n_estimators"])
        * len(search_space["max_depth"])
        * len(search_space["learning_rate"])
        * len(search_space["scaler_name"])
        * len(search_space["balanced_sample_weight"])
    )

    print(f"\nStarting controlled hyperparameter tuning across {total_configs} configurations...", flush=True)
    config_idx = 0

    # Cache scaled datasets to avoid redundant fitting
    scaled_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for s_name in search_space["scaler_name"]:
        scaler = get_scaler(s_name)
        if scaler is not None:
            X_tr = scaler.fit_transform(X_train_raw)
            X_va = scaler.transform(X_val_raw)
        else:
            X_tr, X_va = X_train_raw, X_val_raw
        scaled_cache[str(s_name)] = (X_tr, X_va)

    for n_est in search_space["n_estimators"]:
        for depth in search_space["max_depth"]:
            for lr in search_space["learning_rate"]:
                for s_name in search_space["scaler_name"]:
                    X_tr, X_va = scaled_cache[str(s_name)]
                    for balanced in search_space["balanced_sample_weight"]:
                        config_idx += 1
                        params = {
                            "n_estimators": n_est,
                            "max_depth": depth,
                            "learning_rate": lr,
                            "subsample": 0.8,
                            "colsample_bytree": 0.8,
                            "min_child_weight": 1,
                        }

                        model = build_classifier(params)
                        weights = make_sample_weights(y_train, balanced)
                        t0 = time.perf_counter()
                        model.fit(X_tr, y_train, sample_weight=weights)
                        preds = model.predict(X_va)
                        fit_time = time.perf_counter() - t0

                        val_acc = accuracy_score(y_val, preds)
                        val_macro_f1 = f1_score(y_val, preds, average="macro", zero_division=0)
                        val_weighted_f1 = f1_score(y_val, preds, average="weighted", zero_division=0)

                        record = {
                            **params,
                            "scaler_name": str(s_name),
                            "balanced_sample_weight": balanced,
                            "validation_accuracy": val_acc,
                            "validation_macro_f1": val_macro_f1,
                            "validation_weighted_f1": val_weighted_f1,
                            "fit_seconds": fit_time,
                        }
                        records.append(record)

                        print(
                            f"[{config_idx:02d}/{total_configs}] depth={depth} n_est={n_est} lr={lr} "
                            f"scaler={s_name} balanced={balanced} | "
                            f"Val Acc={val_acc:.4f} Macro F1={val_macro_f1:.4f} ({fit_time:.1f}s)",
                            flush=True,
                        )

    tuning_df = pd.DataFrame(records).sort_values(
        ["validation_macro_f1", "validation_accuracy"], ascending=False
    ).reset_index(drop=True)

    best_row = tuning_df.iloc[0].to_dict()
    best_params = {
        "n_estimators": int(best_row["n_estimators"]),
        "max_depth": int(best_row["max_depth"]),
        "learning_rate": float(best_row["learning_rate"]),
        "subsample": float(best_row["subsample"]),
        "colsample_bytree": float(best_row["colsample_bytree"]),
        "min_child_weight": int(best_row["min_child_weight"]),
    }
    best_balanced = bool(best_row["balanced_sample_weight"])
    best_scaler_name = None if best_row["scaler_name"] == "None" else str(best_row["scaler_name"])

    print("\nOptimal hyperparameter configuration selected by Validation Macro F1:")
    print(f"  - Parameters        : {best_params}")
    print(f"  - Scaler            : {best_scaler_name}")
    print(f"  - Class Weighting   : {best_balanced}")
    print(f"  - Val Accuracy      : {best_row['validation_accuracy']:.4f}")
    print(f"  - Val Macro F1      : {best_row['validation_macro_f1']:.4f}")

    return best_params, best_balanced, best_scaler_name, tuning_df


def evaluate_final_model(
    model: XGBClassifier,
    scaler: Any,
    frames: Dict[str, pd.DataFrame],
    feature_cols: Sequence[str],
) -> Dict[str, Any]:
    """
    Evaluate the final trained hybrid model on the untouched test set.
    """
    y_test = frames["test"]["class_id"].to_numpy(dtype=int)
    X_test_raw = frames["test"][list(feature_cols)].to_numpy(dtype=np.float64)

    if scaler is not None:
        X_test = scaler.transform(X_test_raw)
    else:
        X_test = X_test_raw

    t0 = time.perf_counter()
    preds = model.predict(X_test)
    elapsed = time.perf_counter() - t0

    labels = sorted(CLASS_NAMES)
    cm = confusion_matrix(y_test, preds, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm_cm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    per_class_df = pd.DataFrame({
        "class_id": labels,
        "class_name": [CLASS_NAMES[i] for i in labels],
        "precision": precision_score(y_test, preds, labels=labels, average=None, zero_division=0),
        "recall": recall_score(y_test, preds, labels=labels, average=None, zero_division=0),
        "f1_score": f1_score(y_test, preds, labels=labels, average=None, zero_division=0),
        "support": np.bincount(y_test, minlength=len(labels)),
    })

    acc = accuracy_score(y_test, preds)
    macro_p = precision_score(y_test, preds, average="macro", zero_division=0)
    macro_r = recall_score(y_test, preds, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
    report_str = classification_report(
        y_test,
        preds,
        labels=labels,
        target_names=[CLASS_NAMES[i] for i in labels],
        digits=4,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "normalized_confusion_matrix": norm_cm,
        "per_class": per_class_df,
        "classification_report": report_str,
        "inference_seconds": elapsed,
        "milliseconds_per_image": (elapsed / len(y_test)) * 1000.0,
    }


def analyze_feature_importance(
    model: XGBClassifier,
    feature_cols: Sequence[str],
    feature_groups: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze model-based feature importances across individual features and feature domains.
    """
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"feature": feature_cols, "importance": importances})

    # Map each feature to its group
    col_to_group = {}
    for group_name, cols in feature_groups.items():
        for c in cols:
            col_to_group[c] = group_name

    feat_df["feature_group"] = feat_df["feature"].map(col_to_group)
    feat_df.sort_values("importance", ascending=False, inplace=True)

    # Top 20 features
    top20_df = feat_df.head(20).reset_index(drop=True)

    # Group-level aggregation
    group_summary = []
    total_imp = feat_df["importance"].sum()

    for group_name, cols in feature_groups.items():
        grp_sub = feat_df[feat_df["feature_group"] == group_name]
        grp_sum = grp_sub["importance"].sum()
        grp_mean = grp_sub["importance"].mean()
        top20_count = (top20_df["feature_group"] == group_name).sum()

        group_summary.append({
            "feature_group": group_name,
            "feature_count": len(cols),
            "total_importance": grp_sum,
            "relative_importance_percent": (grp_sum / total_imp) * 100.0 if total_imp > 0 else 0.0,
            "mean_importance": grp_mean,
            "top20_count": top20_count,
        })

    group_summary_df = pd.DataFrame(group_summary).sort_values("total_importance", ascending=False).reset_index(drop=True)
    return top20_df, group_summary_df


def create_plots(
    test_eval: Dict[str, Any],
    tuning_df: pd.DataFrame,
    top20_df: pd.DataFrame,
    group_summary_df: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Generate all 6 required diagnostic visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")
    labels = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]

    # 1. Confusion Matrix
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        test_eval["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_title("Hybrid Model (1342 Features + XGBoost): Test Confusion Matrix", fontsize=12, pad=12)
    ax.set_xlabel("Predicted Class", fontsize=10)
    ax.set_ylabel("True Class", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close(fig)

    # 2. Normalized Confusion Matrix
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        test_eval["normalized_confusion_matrix"] * 100.0,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"label": "Recall (%)"},
    )
    ax.set_title("Hybrid Model (1342 Features + XGBoost): Normalized Test Confusion Matrix (%)", fontsize=12, pad=12)
    ax.set_xlabel("Predicted Class", fontsize=10)
    ax.set_ylabel("True Class", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "normalized_confusion_matrix.png", dpi=300)
    plt.close(fig)

    # 3. Per-Class Metrics
    per_class_df = test_eval["per_class"]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    width = 0.25
    ax.bar(x - width, per_class_df["precision"], width, label="Precision", color="#4C72B0")
    ax.bar(x, per_class_df["recall"], width, label="Recall", color="#55A868")
    ax.bar(x + width, per_class_df["f1_score"], width, label="F1-Score", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Hybrid Model: Per-Class Performance Metrics (Test Set)", fontsize=12, pad=12)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "per_class_metrics.png", dpi=300)
    plt.close(fig)

    # 4. Model Comparison (Ablation Study Chart)
    models = ["Handcrafted + SVM", "EfficientNet + XGBoost", "Hybrid (1342) + XGBoost"]
    accuracies = [BASELINE_SVM["accuracy"] * 100, BASELINE_EFFICIENTNET["accuracy"] * 100, test_eval["accuracy"] * 100]
    macro_f1s = [BASELINE_SVM["macro_f1"] * 100, BASELINE_EFFICIENTNET["macro_f1"] * 100, test_eval["macro_f1"] * 100]
    weighted_f1s = [BASELINE_SVM["weighted_f1"] * 100, BASELINE_EFFICIENTNET["weighted_f1"] * 100, test_eval["weighted_f1"] * 100]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.25
    rects1 = ax.bar(x - width, accuracies, width, label="Accuracy (%)", color="#8175C7")
    rects2 = ax.bar(x, macro_f1s, width, label="Macro F1 (%)", color="#64B5CD")
    rects3 = ax.bar(x + width, weighted_f1s, width, label="Weighted F1 (%)", color="#CCB974")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(75, 100)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Ablation Study: Baseline Models vs Proposed Hybrid Model (Test Set)", fontsize=12, pad=12)
    ax.legend(loc="lower left")

    # Add text annotations on top of bars
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=300)
    plt.close(fig)

    # 5. Top 20 Feature Importances
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(
        data=top20_df,
        x="importance",
        y="feature",
        hue="feature_group",
        dodge=False,
        palette="crest",
        ax=ax,
    )
    ax.set_title("Top 20 Model-Based Feature Importances (Hybrid XGBoost)", fontsize=12, pad=12)
    ax.set_xlabel("XGBoost Importance Score")
    ax.set_ylabel("Feature Name")
    fig.tight_layout()
    fig.savefig(output_dir / "top20_feature_importance.png", dpi=300)
    plt.close(fig)

    # 6. Feature Group Importance Breakdown
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=group_summary_df,
        x="feature_group",
        y="relative_importance_percent",
        palette="viridis",
        ax=ax,
    )
    ax.set_title("Relative Feature Domain Importance Share (%)", fontsize=12, pad=12)
    ax.set_xlabel("Feature Domain")
    ax.set_ylabel("Share of Total Model Importance (%)")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}%",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "feature_group_importance.png", dpi=300)
    plt.close(fig)

    print("All 6 visualization plots saved to:", output_dir)


def generate_report(
    best_params: Dict[str, Any],
    best_balanced: bool,
    best_scaler_name: str | None,
    val_metrics: Dict[str, float],
    test_eval: Dict[str, Any],
    tuning_df: pd.DataFrame,
    top20_df: pd.DataFrame,
    group_summary_df: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Generate comprehensive txt report detailing all 18 specified sections."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "hybrid_model_report.txt"

    acc = test_eval["accuracy"]
    macro_f1 = test_eval["macro_f1"]
    weighted_f1 = test_eval["weighted_f1"]

    acc_diff_svm = (acc - BASELINE_SVM["accuracy"]) * 100
    f1_diff_svm = (macro_f1 - BASELINE_SVM["macro_f1"]) * 100

    acc_diff_eff = (acc - BASELINE_EFFICIENTNET["accuracy"]) * 100
    f1_diff_eff = (macro_f1 - BASELINE_EFFICIENTNET["macro_f1"]) * 100

    improved_over_svm = macro_f1 > BASELINE_SVM["macro_f1"]
    improved_over_eff = macro_f1 > BASELINE_EFFICIENTNET["macro_f1"]

    unweighted_best_val = tuning_df[~tuning_df["balanced_sample_weight"]]["validation_macro_f1"].max()
    weighted_best_val = tuning_df[tuning_df["balanced_sample_weight"]]["validation_macro_f1"].max()

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  MAIN HYBRID FEATURE-FUSION MODEL EVALUATION REPORT\n")
        f.write("  Project: Rice Quality and Defect Assessment using Hybrid ML & Deep Fusion\n")
        f.write(f"  Date   : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. RESEARCH MOTIVATION\n")
        f.write("  Experiment 1 (62 Handcrafted -> SVM) achieved 91.16% Test Accuracy and 0.8494 Macro F1.\n")
        f.write("  Experiment 2 (1280 EfficientNet-B0 -> XGBoost) achieved 90.65% Test Accuracy and 0.8399 Macro F1.\n")
        f.write("  The deep-feature model did NOT outperform the handcrafted baseline alone. This provides direct\n")
        f.write("  empirical motivation for testing complementary feature fusion: combining handcrafted shape, texture,\n")
        f.write("  and color representations with deep visual embeddings to evaluate if feature fusion boosts performance.\n\n")

        f.write("2. HANDCRAFTED FEATURE DIMENSION\n")
        f.write("  Total: 62 features (14 Shape, 12 GLCM Texture, 36 Colour [RGB/HSV/LAB]).\n\n")

        f.write("3. EFFICIENTNET FEATURE DIMENSION\n")
        f.write("  Total: 1280 deep features extracted from pre-trained EfficientNet-B0 backbone.\n\n")

        f.write("4. FINAL HYBRID DIMENSION\n")
        f.write("  Total Fused Vector: 1342 numerical features per image.\n\n")

        f.write("5. DATASET DISTRIBUTION & ALIGNMENT\n")
        f.write("  Splits (aligned strictly via image_path & metadata keys):\n")
        f.write("    - Train      : 24,767 images\n")
        f.write("    - Validation :  3,095 images\n")
        f.write("    - Test       :  3,100 images\n\n")

        f.write("6. PREPROCESSING & FEATURE SCALING\n")
        f.write(f"  Scaling Method Fit on Train Data Only : {best_scaler_name if best_scaler_name else 'None (Raw Features)'}\n")
        f.write("  Leakage Guard: Any scaling transformations were fit strictly on train split and applied to val/test.\n\n")

        f.write("7. XGBOOST CONFIGURATION\n")
        f.write("  Objective: multi:softprob (8 classes), tree_method: hist, eval_metric: mlogloss, random_state: 42.\n\n")

        f.write("8. HYPERPARAMETERS TESTED\n")
        f.write("  - n_estimators           : [100]\n")
        f.write("  - max_depth             : [4]\n")
        f.write("  - learning_rate          : [0.1]\n")
        f.write("  - scalers                : [StandardScaler]\n")
        f.write("  - balanced_sample_weight : [False, True]\n\n")

        f.write("9. BEST HYPERPARAMETERS (Selected by Validation Macro F1)\n")
        f.write(f"  - Parameters             : {best_params}\n")
        f.write(f"  - Feature Scaler         : {best_scaler_name}\n")
        f.write(f"  - Balanced Sample Weights: {best_balanced}\n\n")

        f.write("10. VALIDATION PERFORMANCE\n")
        f.write(f"  - Accuracy   : {val_metrics['accuracy'] * 100:.2f}%\n")
        f.write(f"  - Macro F1   : {val_metrics['macro_f1']:.4f}\n")
        f.write(f"  - Weighted F1: {val_metrics['weighted_f1']:.4f}\n\n")

        f.write("11. FINAL TEST PERFORMANCE (Untouched Test Split)\n")
        f.write(f"  - Test Accuracy   : {test_eval['accuracy'] * 100:.2f}%\n")
        f.write(f"  - Macro Precision : {test_eval['macro_precision'] * 100:.2f}%\n")
        f.write(f"  - Macro Recall    : {test_eval['macro_recall'] * 100:.2f}%\n")
        f.write(f"  - Macro F1        : {test_eval['macro_f1']:.4f}\n")
        f.write(f"  - Weighted F1     : {test_eval['weighted_f1']:.4f}\n")
        f.write(f"  - Inference Latency: {test_eval['inference_seconds']:.4f} s total ({test_eval['milliseconds_per_image']:.4f} ms/image)\n\n")

        f.write("12. PER-CLASS PERFORMANCE BREAKDOWN\n")
        f.write(test_eval["classification_report"] + "\n\n")

        f.write("13. CLASS IMBALANCE ANALYSIS\n")
        f.write(f"  - Unweighted Hybrid Best Val Macro F1 : {unweighted_best_val:.4f}\n")
        f.write(f"  - Weighted Hybrid Best Val Macro F1   : {weighted_best_val:.4f}\n")
        if best_balanced:
            f.write("  - Selection: Balanced sample weighting improved validation Macro F1 and was selected.\n\n")
        else:
            f.write("  - Selection: Unweighted training achieved higher validation Macro F1 and was selected.\n\n")

        f.write("14. FEATURE IMPORTANCE ANALYSIS (Model-Based)\n")
        f.write("  Top 20 Individual Features:\n")
        for idx, row in top20_df.iterrows():
            f.write(f"    {idx+1:02d}. {row['feature']:<30} [{row['feature_group']}] : {row['importance']:.6f}\n")
        f.write("\n  Feature Domain Summary:\n")
        for idx, row in group_summary_df.iterrows():
            f.write(
                f"    - {row['feature_group']:<18} ({row['feature_count']} features): "
                f"Total Share = {row['relative_importance_percent']:.2f}%, "
                f"Mean = {row['mean_importance']:.6f}, Top20 Count = {row['top20_count']}\n"
            )
        f.write("\n")

        f.write("15. ABLATION STUDY & MODEL COMPARISON\n")
        f.write("  " + "-" * 75 + "\n")
        f.write("  Experiment                     Features     Classifier    Accuracy   Macro F1   Weighted F1\n")
        f.write("  " + "-" * 75 + "\n")
        f.write(f"  1. Handcrafted Baseline        62           RBF SVM       {BASELINE_SVM['accuracy']*100:.2f}%     {BASELINE_SVM['macro_f1']:.4f}     {BASELINE_SVM['weighted_f1']:.4f}\n")
        f.write(f"  2. EfficientNet Deep Baseline  1280         XGBoost       {BASELINE_EFFICIENTNET['accuracy']*100:.2f}%     {BASELINE_EFFICIENTNET['macro_f1']:.4f}     {BASELINE_EFFICIENTNET['weighted_f1']:.4f}\n")
        f.write(f"  3. Proposed Hybrid Model       1342         XGBoost       {acc*100:.2f}%     {macro_f1:.4f}     {weighted_f1:.4f}\n")
        f.write("  " + "-" * 75 + "\n\n")

        f.write("16. RESEARCH INTERPRETATION\n")
        f.write(f"  Did Hybrid Feature Fusion improve over Handcrafted SVM baseline? {'YES' if improved_over_svm else 'NO'}\n")
        f.write(f"    - Accuracy Change : {acc_diff_svm:+.2f} percentage points\n")
        f.write(f"    - Macro F1 Change : {f1_diff_svm:+.4f}\n")
        f.write(f"  Did Hybrid Feature Fusion improve over Deep EfficientNet baseline? {'YES' if improved_over_eff else 'NO'}\n")
        f.write(f"    - Accuracy Change : {acc_diff_eff:+.2f} percentage points\n")
        f.write(f"    - Macro F1 Change : {f1_diff_eff:+.4f}\n\n")

        f.write("17. LIMITATIONS\n")
        f.write("  - Dimensionality: Concatenating 1280 deep features with 62 handcrafted features creates a high 1342-D space.\n")
        f.write("  - Feature Redundancy: Deep embeddings may subsume certain texture/color signals, or noise in deep dimensions may obscure handcrafted features.\n\n")

        f.write("18. RECOMMENDED NEXT EXPERIMENT\n")
        if improved_over_svm and improved_over_eff:
            f.write("  Feature fusion successfully improved overall quality and defect classification performance. Next step: explore explicit Feature Selection (e.g. Mutual Information / Lasso / Boruta) to prune redundant deep feature dimensions.\n")
        else:
            f.write("  Hybrid concatenation provided modest/mixed gains. Next experiment should test explicit Feature Selection / Dimensionality Reduction (e.g., PCA/SelectKBest on deep features before fusion) or LightGBM/CatBoost architectures to optimize hybrid representation efficiency.\n")

    print(f"Comprehensive hybrid model report saved to: {report_path}")


def save_required_artifacts(
    model: XGBClassifier,
    scaler: Any,
    best_params: Dict[str, Any],
    best_balanced: bool,
    best_scaler_name: str | None,
    val_metrics: Dict[str, float],
    test_eval: Dict[str, Any],
    tuning_df: pd.DataFrame,
    group_summary_df: pd.DataFrame,
    final_fit_seconds: float,
) -> None:
    """Persist the requested reproducibility artifacts using stable, explicit names."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUTPUT_DIR / "hybrid_xgboost_model.joblib")
    joblib.dump(scaler, OUTPUT_DIR / "hybrid_scaler.joblib")
    with (OUTPUT_DIR / "hybrid_best_params.json").open("w", encoding="utf-8") as handle:
        json.dump({**best_params, "scaler_name": best_scaler_name, "balanced_sample_weight": best_balanced,
                   "random_state": RANDOM_STATE}, handle, indent=2)
    with (OUTPUT_DIR / "hybrid_validation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(val_metrics, handle, indent=2)
    serializable_test = {key: value for key, value in test_eval.items()
                         if key not in {"confusion_matrix", "normalized_confusion_matrix", "per_class", "classification_report"}}
    with (OUTPUT_DIR / "hybrid_test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(serializable_test, handle, indent=2)
    test_eval["per_class"].to_csv(OUTPUT_DIR / "hybrid_per_class_metrics.csv", index=False)
    pd.DataFrame(test_eval["confusion_matrix"], index=[CLASS_NAMES[i] for i in CLASS_NAMES],
                 columns=[CLASS_NAMES[i] for i in CLASS_NAMES]).to_csv(OUTPUT_DIR / "hybrid_confusion_matrix.csv")
    group_summary_df.to_csv(OUTPUT_DIR / "hybrid_feature_group_importance.csv", index=False)
    tuning_df.to_csv(OUTPUT_DIR / "hybrid_tuning_results.csv", index=False)
    comparison = pd.DataFrame([
        {"model": "SVM + handcrafted features", "accuracy": 0.9116, "macro_precision": 0.8206,
         "macro_recall": 0.8860, "macro_f1": 0.8494, "weighted_f1": 0.9161},
        {"model": "EfficientNet-B0 features + XGBoost", "accuracy": 0.9065, "macro_precision": 0.8180,
         "macro_recall": 0.8679, "macro_f1": 0.8399, "weighted_f1": 0.9099},
        {"model": "Hybrid handcrafted + EfficientNet-B0 + XGBoost", "accuracy": test_eval["accuracy"],
         "macro_precision": test_eval["macro_precision"], "macro_recall": test_eval["macro_recall"],
         "macro_f1": test_eval["macro_f1"], "weighted_f1": test_eval["weighted_f1"]},
    ])
    comparison.to_csv(OUTPUT_DIR / "hybrid_model_comparison.csv", index=False)
    for source, target in [("confusion_matrix.png", "hybrid_confusion_matrix.png"),
                           ("feature_group_importance.png", "hybrid_feature_group_importance.png"),
                           ("model_comparison.png", "hybrid_model_comparison.png"),
                           ("hybrid_model_report.txt", "hybrid_experiment_report.txt")]:
        shutil.copy2(OUTPUT_DIR / source, OUTPUT_DIR / target)
    with (OUTPUT_DIR / "hybrid_reproducibility.json").open("w", encoding="utf-8") as handle:
        json.dump({"random_state": RANDOM_STATE, "sample_counts": EXPECTED_COUNTS, "feature_dimension": 1342,
                   "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                   "pandas": pd.__version__, "tuning_fit_seconds": float(tuning_df["fit_seconds"].sum()),
                   "final_fit_seconds": final_fit_seconds}, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Main Hybrid XGBoost Model.")
    parser.add_argument("--save-hybrid-csv", action="store_true", default=True, help="Save merged hybrid CSV features")
    args = parser.parse_args()

    # 1. Load & Align Features
    frames, feature_cols, feature_groups = load_and_align_features()

    # 2. Save Hybrid Feature Tables
    if args.save_hybrid_csv:
        save_hybrid_features(frames)

    # 3. Hyperparameter Tuning & Class Imbalance Search
    best_params, best_balanced, best_scaler_name, tuning_df = tune_hybrid_model(frames, feature_cols)

    best_val_row = tuning_df.iloc[0]
    val_metrics = {
        "accuracy": float(best_val_row["validation_accuracy"]),
        "macro_f1": float(best_val_row["validation_macro_f1"]),
        "weighted_f1": float(best_val_row["validation_weighted_f1"]),
    }

    # 4. Fit Final Model on Training Set with Best Config
    print("\nFitting final model on full training set with optimal hyperparameters...")
    scaler = get_scaler(best_scaler_name)
    X_train_raw = frames["train"][feature_cols].to_numpy(dtype=np.float64)
    y_train = frames["train"]["class_id"].to_numpy(dtype=int)

    if scaler is not None:
        X_train = scaler.fit_transform(X_train_raw)
    else:
        X_train = X_train_raw

    final_model = build_classifier(best_params)
    weights = make_sample_weights(y_train, best_balanced)
    final_fit_started = time.perf_counter()
    final_model.fit(X_train, y_train, sample_weight=weights)
    final_fit_seconds = time.perf_counter() - final_fit_started

    # 5. Evaluate Final Model on Untouched Test Set
    print("\nEvaluating final model on untouched test set...")
    test_eval = evaluate_final_model(final_model, scaler, frames, feature_cols)

    comparison = pd.DataFrame([
        {
            "Model": "SVM + Handcrafted Features",
            "Feature Type": "62 handcrafted",
            "Accuracy": BASELINE_SVM["accuracy"],
            "Macro F1": BASELINE_SVM["macro_f1"],
            "Weighted F1": BASELINE_SVM["weighted_f1"],
        },
        {
            "Model": "XGBoost + EfficientNet Features",
            "Feature Type": "1280 EfficientNet-B0 deep",
            "Accuracy": BASELINE_EFFICIENTNET["accuracy"],
            "Macro F1": BASELINE_EFFICIENTNET["macro_f1"],
            "Weighted F1": BASELINE_EFFICIENTNET["weighted_f1"],
        },
        {
            "Model": "XGBoost + Hybrid Features",
            "Feature Type": "1342 fused",
            "Accuracy": test_eval["accuracy"],
            "Macro F1": test_eval["macro_f1"],
            "Weighted F1": test_eval["weighted_f1"],
        },
    ])
    comparison["Accuracy Change vs SVM"] = comparison["Accuracy"] - BASELINE_SVM["accuracy"]
    comparison["Macro F1 Change vs SVM"] = comparison["Macro F1"] - BASELINE_SVM["macro_f1"]
    comparison.to_csv(PROJECT_ROOT / "results" / "models" / "model_comparison_final.csv", index=False)

    # 6. Feature Contribution Analysis
    top20_df, group_summary_df = analyze_feature_importance(final_model, feature_cols, feature_groups)

    # 7. Generate Visualizations
    create_plots(test_eval, tuning_df, top20_df, group_summary_df)

    # 8. Save Model Pipeline & Best Parameters
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "scaler": scaler, "feature_columns": feature_cols}, OUTPUT_DIR / "hybrid_xgboost.joblib")
    with open(OUTPUT_DIR / "best_parameters.json", "w", encoding="utf-8") as f:
        json.dump({**best_params, "scaler_name": best_scaler_name, "balanced_sample_weight": best_balanced}, f, indent=4)

    # 9. Write Comprehensive Report
    generate_report(best_params, best_balanced, best_scaler_name, val_metrics, test_eval, tuning_df, top20_df, group_summary_df)
    save_required_artifacts(final_model, scaler, best_params, best_balanced, best_scaler_name,
                            val_metrics, test_eval, tuning_df, group_summary_df, final_fit_seconds)

    # 10. Print Summary to Console
    print("\n" + "=" * 60)
    print("HYBRID MODEL EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Hybrid Feature Dimension : {len(feature_cols)} (62 Handcrafted + 1280 Deep)")
    print(f"Best Hyperparameters     : {best_params}")
    print(f"Best Feature Scaler      : {best_scaler_name}")
    print(f"Class Weighting Selected : {best_balanced}")
    print(f"Validation Accuracy      : {val_metrics['accuracy']*100:.2f}%")
    print(f"Validation Macro F1      : {val_metrics['macro_f1']:.4f}")
    print(f"Test Accuracy            : {test_eval['accuracy']*100:.2f}%")
    print(f"Test Macro F1            : {test_eval['macro_f1']:.4f}")
    print(f"Test Weighted F1         : {test_eval['weighted_f1']:.4f}")
    print("-" * 60)
    print("Comparison with Previous Baseline Models:")
    print(f"  Handcrafted SVM       : Accuracy = {BASELINE_SVM['accuracy']*100:.2f}%, Macro F1 = {BASELINE_SVM['macro_f1']:.4f}")
    print(f"  EfficientNet XGBoost  : Accuracy = {BASELINE_EFFICIENTNET['accuracy']*100:.2f}%, Macro F1 = {BASELINE_EFFICIENTNET['macro_f1']:.4f}")
    print(f"  Main Hybrid XGBoost   : Accuracy = {test_eval['accuracy']*100:.2f}%, Macro F1 = {test_eval['macro_f1']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
