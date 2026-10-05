"""
train_svm_baseline.py
=====================
Baseline SVM Model Training, Validation Tuning, and Test Evaluation

Project:
    Rice Quality and Defect Assessment using Hybrid Machine Learning and Deep Feature Fusion

Description:
    Trains and evaluates a conventional baseline Support Vector Machine (SVM) classifier
    using 62 handcrafted features (14 Shape, 12 GLCM Texture, 36 Colour).

Workflow:
    1. Loads train, val, test split feature CSVs (no data leakage).
    2. Constructs a scikit-learn Pipeline with StandardScaler and RBF SVC.
    3. Performs hyperparameter tuning (C, gamma, class_weight) on the validation split.
    4. Evaluates the best model on the untouched test split.
    5. Saves the trained pipeline to results/models/svm/svm_baseline.joblib.
    6. Generates 4 rich visual evaluations in results/models/svm/.
    7. Generates results/models/svm/svm_baseline_report.txt.

Important Research Note:
    This model serves strictly as the conventional BASELINE to establish a benchmark
    for subsequent comparisons against XGBoost, EfficientNet-B0, and the proposed Hybrid Fusion model.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Project Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_DIR = PROJECT_ROOT / "results" / "features"
OUTPUT_DIR = PROJECT_ROOT / "results" / "models" / "svm"

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

META_COLUMNS = ["image_path", "split", "class_id", "class_name"]


# ---------------------------------------------------------------------------
# 1. DATA LOADING & PREPARATION
# ---------------------------------------------------------------------------
def load_feature_data(
    features_dir: Path = FEATURES_DIR
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load train, validation, and test feature CSV files.

    Returns:
        train_df, val_df, test_df, feature_names
    """
    train_csv = features_dir / "train_handcrafted_features.csv"
    val_csv = features_dir / "val_handcrafted_features.csv"
    test_csv = features_dir / "test_handcrafted_features.csv"

    for p in [train_csv, val_csv, test_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Required feature CSV not found: {p}")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    feature_cols = [c for c in train_df.columns if c not in META_COLUMNS]
    if len(feature_cols) != 62:
        raise ValueError(f"Expected 62 handcrafted features, found {len(feature_cols)}")
    for split_name, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        missing = set(feature_cols) - set(split_df.columns)
        if missing:
            raise ValueError(f"{split_name} split is missing feature columns: {sorted(missing)}")
        if not np.isfinite(split_df[feature_cols].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{split_name} split contains NaN or infinite feature values")

    print(f"[*] Loaded datasets from {features_dir}:")
    print(f"    Train Split : {len(train_df):,} samples x {len(feature_cols)} features")
    print(f"    Val Split   : {len(val_df):,} samples x {len(feature_cols)} features")
    print(f"    Test Split  : {len(test_df):,} samples x {len(feature_cols)} features")

    return train_df, val_df, test_df, feature_cols


def prepare_tensors(
    df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Extract feature matrix X, target vector y, and metadata DataFrame.
    """
    X = df[feature_cols].values.astype(np.float64)
    y = df["class_id"].values.astype(np.int64)
    meta_cols = [c for c in df.columns if c in META_COLUMNS]
    meta_df = df[meta_cols].copy()
    return X, y, meta_df


# ---------------------------------------------------------------------------
# 2. MODEL PIPELINE CREATION
# ---------------------------------------------------------------------------
def build_svm_pipeline(
    C: float = 1.0,
    gamma: Union[str, float] = "scale",
    class_weight: Optional[str] = None,
    kernel: str = "rbf",
    probability: bool = True,
    random_state: int = 42,
) -> Pipeline:
    """
    Build scikit-learn Pipeline with StandardScaler and RBF SVC.
    Ensures zero data leakage: scaler is fit only on training data.
    """
    scaler = StandardScaler()
    svm = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        class_weight=class_weight,
        probability=probability,
        random_state=random_state,
    )
    pipeline = Pipeline([
        ("scaler", scaler),
        ("svm", svm),
    ])
    return pipeline


# ---------------------------------------------------------------------------
# 3. HYPERPARAMETER TUNING ON VALIDATION SET
# ---------------------------------------------------------------------------
def tune_svm_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    param_grid: Optional[Dict[str, List[Any]]] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Perform controlled grid search across C, gamma, and class_weight.
    Evaluates each candidate model on the validation split.

    Selection criterion: Validation Macro F1 (optimal for class-imbalanced multi-class).
    """
    if param_grid is None:
        param_grid = {
            "C": [0.1, 1.0, 5.0, 10.0, 50.0],
            "gamma": ["scale", "auto", 0.01, 0.05, 0.1],
            "class_weight": [None, "balanced"],
        }

    print("\n" + "=" * 70)
    print("  HYPERPARAMETER TUNING ON VALIDATION SPLIT")
    print("=" * 70)
    print(f"  Candidate C values        : {param_grid['C']}")
    print(f"  Candidate gamma values    : {param_grid['gamma']}")
    print(f"  Candidate class_weights   : {param_grid['class_weight']}")
    print("=" * 70)

    tuning_records = []
    total_configs = len(param_grid["C"]) * len(param_grid["gamma"]) * len(param_grid["class_weight"])
    idx = 0

    for cw in param_grid["class_weight"]:
        for c in param_grid["C"]:
            for g in param_grid["gamma"]:
                idx += 1
                t0 = time.perf_counter()
                
                # Build and fit pipeline strictly on train data
                pipe = build_svm_pipeline(C=c, gamma=g, class_weight=cw, probability=False)
                pipe.fit(X_train, y_train)
                train_time = time.perf_counter() - t0

                # Evaluate on validation split
                t_val_start = time.perf_counter()
                val_preds = pipe.predict(X_val)
                val_inf_time = time.perf_counter() - t_val_start

                val_acc = accuracy_score(y_val, val_preds)
                val_macro_f1 = f1_score(y_val, val_preds, average="macro")
                val_weighted_f1 = f1_score(y_val, val_preds, average="weighted")
                val_macro_prec = precision_score(y_val, val_preds, average="macro", zero_division=0)
                val_macro_rec = recall_score(y_val, val_preds, average="macro", zero_division=0)

                rec = {
                    "config_id": idx,
                    "C": c,
                    "gamma": str(g),
                    "class_weight": str(cw),
                    "train_time_s": train_time,
                    "val_inf_time_s": val_inf_time,
                    "val_accuracy": val_acc,
                    "val_macro_f1": val_macro_f1,
                    "val_weighted_f1": val_weighted_f1,
                    "val_macro_precision": val_macro_prec,
                    "val_macro_recall": val_macro_rec,
                }
                tuning_records.append(rec)

                print(f"  [{idx:02d}/{total_configs:02d}] C={c:<5} gamma={str(g):<6s} cw={str(cw):<8s} | Val Acc: {val_acc*100:5.2f}% | Val Macro F1: {val_macro_f1:.4f} | Train Time: {train_time:4.1f}s")

    tuning_df = pd.DataFrame(tuning_records)
    best_row = tuning_df.sort_values(by="val_macro_f1", ascending=False).iloc[0]

    best_params = {
        "C": best_row["C"],
        "gamma": "scale" if best_row["gamma"] == "scale" else ("auto" if best_row["gamma"] == "auto" else float(best_row["gamma"])),
        "class_weight": None if best_row["class_weight"] == "None" else "balanced",
    }

    print("\n" + "=" * 70)
    print("  BEST HYPERPARAMETER CONFIGURATION IDENTIFIED")
    print("=" * 70)
    print(f"  Best C            : {best_params['C']}")
    print(f"  Best gamma        : {best_params['gamma']}")
    print(f"  Best class_weight : {best_params['class_weight']}")
    print(f"  Validation Accuracy : {best_row['val_accuracy']*100:.2f}%")
    print(f"  Validation Macro F1 : {best_row['val_macro_f1']:.4f}")
    print(f"  Validation Weighted F1 : {best_row['val_weighted_f1']:.4f}")
    print("=" * 70)

    return best_params, tuning_df


# ---------------------------------------------------------------------------
# 4. TEST SET EVALUATION
# ---------------------------------------------------------------------------
def evaluate_on_test_set(
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Dict[int, str]
) -> Dict[str, Any]:
    """
    Perform final comprehensive evaluation of the fitted SVM pipeline on the test set.
    """
    print("\n[*] Evaluating final SVM pipeline on untouched TEST split ...")
    t0 = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    total_inf_time = time.perf_counter() - t0
    num_samples = len(y_test)
    ms_per_sample = (total_inf_time / num_samples) * 1000.0

    acc = accuracy_score(y_test, y_pred)
    macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    weighted_prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    weighted_rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    row_totals = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(np.float64), row_totals, out=np.zeros_like(cm, dtype=np.float64), where=row_totals != 0)

    # Per-class metrics
    per_class_prec = precision_score(y_test, y_pred, average=None, zero_division=0)
    per_class_rec = recall_score(y_test, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
    per_class_support = np.bincount(y_test, minlength=len(class_names))

    per_class_df = pd.DataFrame({
        "class_id": sorted(class_names.keys()),
        "class_name": [class_names[i] for i in sorted(class_names.keys())],
        "precision": per_class_prec,
        "recall": per_class_rec,
        "f1_score": per_class_f1,
        "support": per_class_support,
    })

    clf_report = classification_report(
        y_test,
        y_pred,
        target_names=[class_names[i] for i in sorted(class_names.keys())],
        digits=4,
        zero_division=0
    )

    return {
        "y_true": y_test,
        "y_pred": y_pred,
        "accuracy": acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_prec,
        "weighted_recall": weighted_rec,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "confusion_matrix_normalized": cm_norm,
        "per_class_metrics": per_class_df,
        "classification_report_str": clf_report,
        "total_inference_time_s": total_inf_time,
        "ms_per_sample": ms_per_sample,
    }


# ---------------------------------------------------------------------------
# 5. VISUALIZATIONS GENERATOR
# ---------------------------------------------------------------------------
def generate_svm_visualizations(
    eval_results: Dict[str, Any],
    class_names: Dict[int, str],
    tuning_df: pd.DataFrame,
    best_params: Dict[str, Any],
    output_dir: Path,
) -> List[Path]:
    """
    Generate the 4 required baseline SVM visual evaluation plots:
    1. confusion_matrix.png (Raw counts)
    2. normalized_confusion_matrix.png (Percentages)
    3. per_class_metrics.png (Bar chart of Precision, Recall, F1 by class)
    4. model_performance_summary.png (Overview dashboard)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_plots = []

    labels = [class_names[i] for i in sorted(class_names.keys())]
    plt.style.use("dark_background")

    # -----------------------------------------------------------------------
    # 1. Raw Confusion Matrix
    # -----------------------------------------------------------------------
    print("[*] Generating Visualization 1: Raw Confusion Matrix ...")
    fig, ax = plt.subplots(figsize=(10, 8.5), facecolor="#090d16")
    ax.set_facecolor("#0f172a")

    cm = eval_results["confusion_matrix"]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Sample Count"},
        linewidths=0.5,
        linecolor="#1e293b",
        ax=ax,
    )
    plt.title(f"Baseline SVM: Raw Confusion Matrix (Test Set: {len(eval_results['y_true']):,} Images)", fontsize=13, fontweight="bold", color="#f8fafc", pad=12)
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold", color="#e2e8f0", labelpad=10)
    plt.ylabel("True Class", fontsize=11, fontweight="bold", color="#e2e8f0", labelpad=10)
    plt.xticks(rotation=45, ha="right", color="#cbd5e1")
    plt.yticks(rotation=0, color="#cbd5e1")

    cm_path = output_dir / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(str(cm_path), dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    generated_plots.append(cm_path)

    # -----------------------------------------------------------------------
    # 2. Normalized Confusion Matrix
    # -----------------------------------------------------------------------
    print("[*] Generating Visualization 2: Normalized Confusion Matrix ...")
    fig, ax = plt.subplots(figsize=(10, 8.5), facecolor="#090d16")
    ax.set_facecolor("#0f172a")

    cm_norm = eval_results["confusion_matrix_normalized"]
    sns.heatmap(
        cm_norm * 100.0,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Recall (%)"},
        linewidths=0.5,
        linecolor="#1e293b",
        vmin=0,
        vmax=100,
        ax=ax,
    )
    plt.title("Baseline SVM: Normalized Confusion Matrix (% Recall on Test Set)", fontsize=13, fontweight="bold", color="#f8fafc", pad=12)
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold", color="#e2e8f0", labelpad=10)
    plt.ylabel("True Class", fontsize=11, fontweight="bold", color="#e2e8f0", labelpad=10)
    plt.xticks(rotation=45, ha="right", color="#cbd5e1")
    plt.yticks(rotation=0, color="#cbd5e1")

    cm_norm_path = output_dir / "normalized_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(str(cm_norm_path), dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    generated_plots.append(cm_norm_path)

    # -----------------------------------------------------------------------
    # 3. Per-Class Precision, Recall, and F1-Score Bar Chart
    # -----------------------------------------------------------------------
    print("[*] Generating Visualization 3: Per-Class Metrics ...")
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="#090d16")
    ax.set_facecolor("#0f172a")

    pc_df = eval_results["per_class_metrics"]
    x = np.arange(len(labels))
    width = 0.26

    rects1 = ax.bar(x - width, pc_df["precision"] * 100.0, width, label="Precision", color="#38bdf8", alpha=0.9, edgecolor="#ffffff", linewidth=0.3)
    rects2 = ax.bar(x, pc_df["recall"] * 100.0, width, label="Recall", color="#a855f7", alpha=0.9, edgecolor="#ffffff", linewidth=0.3)
    rects3 = ax.bar(x + width, pc_df["f1_score"] * 100.0, width, label="F1-Score", color="#10b981", alpha=0.9, edgecolor="#ffffff", linewidth=0.3)

    # Add value annotations
    for rect in rects3:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.0, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", color="#f8fafc", fontsize=8, fontweight="bold")

    ax.set_title("Baseline SVM: Per-Class Precision, Recall, and F1-Score (Test Set)", fontsize=13, fontweight="bold", color="#f8fafc", pad=12)
    ax.set_xlabel("Rice Grain Class", fontsize=11, fontweight="bold", color="#e2e8f0", labelpad=8)
    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold", color="#e2e8f0", labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="#cbd5e1", fontsize=9.5)
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", framealpha=0.85, facecolor="#1e293b", edgecolor="#334155")
    ax.grid(axis="y", alpha=0.15, linestyle="--")

    per_class_path = output_dir / "per_class_metrics.png"
    plt.tight_layout()
    plt.savefig(str(per_class_path), dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    generated_plots.append(per_class_path)

    # -----------------------------------------------------------------------
    # 4. Model Performance Summary Dashboard
    # -----------------------------------------------------------------------
    print("[*] Generating Visualization 4: Performance Summary Dashboard ...")
    fig = plt.figure(figsize=(16, 10), facecolor="#090d16")
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    # Panel A: Overall Metric Cards
    ax_cards = fig.add_subplot(gs[0, 0])
    ax_cards.set_facecolor("#0f172a")
    ax_cards.axis("off")
    
    metric_items = [
        ("Test Accuracy", f"{eval_results['accuracy']*100:.2f}%", "#38bdf8"),
        ("Macro F1-Score", f"{eval_results['macro_f1']:.4f}", "#10b981"),
        ("Weighted F1-Score", f"{eval_results['weighted_f1']:.4f}", "#a855f7"),
        ("Macro Recall", f"{eval_results['macro_recall']*100:.2f}%", "#f59e0b"),
        ("Macro Precision", f"{eval_results['macro_precision']*100:.2f}%", "#ec4899"),
        ("Inference Latency", f"{eval_results['ms_per_sample']:.2f} ms/img", "#06b6d4"),
    ]
    
    ax_cards.text(0.5, 0.95, "Key Baseline Performance Metrics (Test Set)", ha="center", va="top", color="#f8fafc", fontsize=12, fontweight="bold")
    for k, (m_name, m_val, m_col) in enumerate(metric_items):
        rx = 0.08 + (k % 2) * 0.48
        ry = 0.70 - (k // 2) * 0.28
        ax_cards.text(rx, ry + 0.08, m_name, color="#94a3b8", fontsize=9, fontweight="bold")
        ax_cards.text(rx, ry - 0.04, m_val, color=m_col, fontsize=16, fontweight="bold")

    # Panel B: Hyperparameter Tuning Curve (C vs Val Macro F1)
    ax_tune = fig.add_subplot(gs[0, 1])
    ax_tune.set_facecolor("#0f172a")
    
    sns.lineplot(
        data=tuning_df,
        x="C",
        y="val_macro_f1",
        hue="gamma",
        style="class_weight",
        markers=True,
        dashes=False,
        palette="tab10",
        ax=ax_tune,
    )
    ax_tune.set_xscale("log")
    ax_tune.set_title("Hyperparameter Exploration: C vs Validation Macro F1", color="#38bdf8", fontsize=11, fontweight="bold", pad=8)
    ax_tune.set_xlabel("Regularization Parameter (C) [Log Scale]", color="#94a3b8", fontsize=9)
    ax_tune.set_ylabel("Validation Macro F1", color="#94a3b8", fontsize=9)
    ax_tune.tick_params(colors="#cbd5e1", labelsize=8)
    ax_tune.legend(loc="lower right", fontsize=7.5, framealpha=0.75, facecolor="#1e293b", edgecolor="#334155")
    ax_tune.grid(True, alpha=0.15, linestyle="--")

    # Panel C: Class Support Distribution vs F1 Score
    ax_supp = fig.add_subplot(gs[1, 0])
    ax_supp.set_facecolor("#0f172a")
    
    sns.barplot(data=pc_df, x="class_name", y="support", color="#475569", alpha=0.7, ax=ax_supp)
    ax_supp.set_ylabel("Test Support (Count)", color="#94a3b8", fontsize=9)
    ax_supp.set_xlabel("Rice Class", color="#94a3b8", fontsize=9)
    ax_supp.tick_params(colors="#cbd5e1", labelsize=8, rotation=30)
    ax_supp.set_title("Test Sample Distribution & Class Imbalance", color="#38bdf8", fontsize=11, fontweight="bold", pad=8)
    ax_supp.grid(axis="y", alpha=0.15, linestyle="--")

    # Secondary axis for F1
    ax_f1_line = ax_supp.twinx()
    ax_f1_line.plot(np.arange(len(pc_df)), pc_df["f1_score"] * 100.0, color="#10b981", marker="o", linewidth=2.0, label="F1-Score (%)")
    ax_f1_line.set_ylabel("Class F1-Score (%)", color="#10b981", fontsize=9, fontweight="bold")
    ax_f1_line.set_ylim(0, 105)
    ax_f1_line.tick_params(colors="#10b981", labelsize=8)

    # Panel D: Architecture Summary & Benchmark Role
    ax_info = fig.add_subplot(gs[1, 1])
    ax_info.set_facecolor("#0f172a")
    ax_info.axis("off")

    info_text = (
        "Baseline Model Configuration:\n"
        f"  • Classifier : Support Vector Machine (RBF Kernel)\n"
        f"  • Optimal C  : {best_params['C']}\n"
        f"  • Gamma      : {best_params['gamma']}\n"
        f"  • Class Wt.  : {best_params['class_weight']}\n"
        "  • Pipeline   : StandardScaler -> SVC(RBF)\n"
        "  • Input Dim  : 62 Handcrafted Features\n"
        "      - 14 Shape / Morphological\n"
        "      - 12 GLCM Texture (Grain Only)\n"
        "      - 36 Colour (RGB, HSV, LAB)\n\n"
        "Benchmark Role in Project:\n"
        "  • Conventional ML Baseline to evaluate feature engineering.\n"
        "  • Will serve as comparison standard for:\n"
        "      1. Handcrafted Features + XGBoost\n"
        "      2. EfficientNet-B0 Deep Features\n"
        "      3. Hybrid Deep + Handcrafted Fusion Model\n"
    )
    ax_info.text(0.05, 0.95, "Baseline Benchmark Specification", color="#f8fafc", fontsize=12, fontweight="bold", va="top")
    ax_info.text(0.05, 0.82, info_text, color="#cbd5e1", fontsize=8.5, fontfamily="monospace", va="top", linespacing=1.4)

    fig.suptitle("Rice Grain Defect Assessment: Conventional SVM Baseline Performance Dashboard", color="#f8fafc", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])
    summary_path = output_dir / "model_performance_summary.png"
    plt.savefig(str(summary_path), dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    generated_plots.append(summary_path)

    return generated_plots


# ---------------------------------------------------------------------------
# 6. REPORT GENERATOR
# ---------------------------------------------------------------------------
def generate_svm_report(
    train_size: int,
    val_size: int,
    test_size: int,
    feature_cols: List[str],
    tuning_df: pd.DataFrame,
    best_params: Dict[str, Any],
    val_metrics: Dict[str, float],
    eval_results: Dict[str, Any],
    report_path: Path,
    model_save_path: Path,
) -> None:
    """
    Generate comprehensive baseline SVM audit and performance report.
    """
    shape_cols = [c for c in feature_cols if c.startswith("shape_")]
    texture_cols = [c for c in feature_cols if c.startswith("glcm_")]
    color_cols = [c for c in feature_cols if c.startswith("color_")]

    lines = [
        "=" * 80,
        "  BASELINE SVM MODEL EVALUATION REPORT",
        "  Project: Rice Quality and Defect Assessment using Hybrid ML & Deep Feature Fusion",
        f"  Date   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
        "",
        "1. RESEARCH ROLE & MODEL CONTEXT",
        "  - Model Type: Conventional Support Vector Classifier (RBF Kernel)",
        "  - Purpose   : Establish a strong classical ML baseline using 62 handcrafted features.",
        "  - Note      : This is NOT the proposed final model. It establishes the benchmark standard",
        "                against which Handcrafted+XGBoost, EfficientNet-B0, and the Hybrid Fusion",
        "                classifier will be measured.",
        "",
        "2. DATASET SPLITS & FEATURE COMPOSITION",
        f"  - Training Set Size       : {train_size:,} images",
        f"  - Validation Set Size     : {val_size:,} images (used exclusively for hyperparameter tuning)",
        f"  - Test Set Size           : {test_size:,} images (evaluated exactly once on final model)",
        f"  - Total Features          : {len(feature_cols)} handcrafted features",
        f"    * Shape Features        : {len(shape_cols)} features",
        f"    * GLCM Texture Features : {len(texture_cols)} features",
        f"    * Colour Features       : {len(color_cols)} features (RGB, HSV, LAB)",
        "",
        "3. PREPROCESSING & LEAKAGE PREVENTION",
        "  - Feature Scaling         : StandardScaler (zero mean, unit variance)",
        "  - Pipeline Implementation : sklearn.pipeline.Pipeline([('scaler', StandardScaler()), ('svm', SVC())])",
        "  - Leakage Guard           : StandardScaler was fit STRICTLY on training data.",
        "",
        "4. HYPERPARAMETER TUNING & VALIDATION PERFORMANCE",
        f"  - Hyperparameters Tested  : C in {sorted(tuning_df['C'].unique().tolist())}, gamma in {sorted(tuning_df['gamma'].unique().tolist())}, class_weight in {sorted(tuning_df['class_weight'].unique().tolist())}",
        f"  - Total Configurations    : {len(tuning_df)} evaluated",
        f"  - Optimal Hyperparameters : C = {best_params['C']}, gamma = '{best_params['gamma']}', class_weight = {best_params['class_weight']}",
        f"  - Validation Accuracy     : {val_metrics['val_accuracy']*100:.2f}%",
        f"  - Validation Macro F1     : {val_metrics['val_macro_f1']:.4f}",
        f"  - Validation Weighted F1  : {val_metrics['val_weighted_f1']:.4f}",
        "",
        "5. FINAL TEST SET PERFORMANCE (Untouched Evaluation)",
        f"  - Test Accuracy           : {eval_results['accuracy']*100:.2f}%",
        f"  - Macro Precision         : {eval_results['macro_precision']*100:.2f}%",
        f"  - Macro Recall            : {eval_results['macro_recall']*100:.2f}%",
        f"  - Macro F1-Score          : {eval_results['macro_f1']:.4f} ({eval_results['macro_f1']*100:.2f}%)",
        f"  - Weighted F1-Score       : {eval_results['weighted_f1']:.4f} ({eval_results['weighted_f1']*100:.2f}%)",
        f"  - Total Inference Time    : {eval_results['total_inference_time_s']:.3f} s ({eval_results['ms_per_sample']:.2f} ms / sample)",
        "",
        "6. PER-CLASS CLASSIFICATION BREAKDOWN (Test Set)",
        "  " + "-" * 74,
        f"  {'Class ID':<9} {'Class Name':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}",
        "  " + "-" * 74,
    ]

    pc_df = eval_results["per_class_metrics"]
    for _, row in pc_df.iterrows():
        lines.append(
            f"  {int(row['class_id']):<9} {row['class_name']:<12} {row['precision']*100:6.2f}%     {row['recall']*100:6.2f}%     {row['f1_score']*100:6.2f}%     {int(row['support']):<8}"
        )

    cm = eval_results["confusion_matrix"]
    lines.append("  Predicted: " + " ".join(f"{class_names[i]:>8}" for i in sorted(class_names)))
    for class_id, row in zip(sorted(class_names), cm):
        lines.append(f"  True {class_names[class_id]:<8} " + " ".join(f"{int(value):8d}" for value in row))

    lines.extend([
        "  " + "-" * 74,
        "",
        "7. FULL CLASSIFICATION REPORT",
        eval_results["classification_report_str"],
        "",
        "8. CONFUSION MATRIX ANALYSIS & OBSERVATIONS",
        "  Confusion matrix rows are true classes and columns are predicted classes:",
        "  - Class support and performance should be interpreted together; the table above reports support for each class.",
        "  - The validation-selected C, gamma, and class_weight configuration is used for the final test evaluation.",
        "  - This benchmark does not make claims about the proposed/final model; later models must be compared against it.",
        "",
        "9. SAVED ARTIFACTS",
        f"  - Serialized Pipeline Model : {model_save_path}",
        f"  - Visualizations Directory  : {report_path.parent}",
        "      * confusion_matrix.png",
        "      * normalized_confusion_matrix.png",
        "      * per_class_metrics.png",
        "      * model_performance_summary.png",
        "",
        "=" * 80,
        "  END OF REPORT",
        "=" * 80,
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# MAIN EXECUTION ROUTINE
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate Baseline SVM Model")
    parser.add_argument("--features-dir", type=str, default=str(FEATURES_DIR), help="Path to feature CSVs")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="Output directory for model and results")
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  Rice Quality & Defect Assessment - Baseline SVM Training & Evaluation")
    print(f"  Features Directory : {features_dir}")
    print(f"  Output Directory   : {output_dir}")
    print("=" * 80)

    # 1. Load Data
    train_df, val_df, test_df, feature_cols = load_feature_data(features_dir)
    X_train, y_train, _ = prepare_tensors(train_df, feature_cols)
    X_val, y_val, _ = prepare_tensors(val_df, feature_cols)
    X_test, y_test, _ = prepare_tensors(test_df, feature_cols)

    # 2. Hyperparameter Tuning on Validation Split
    best_params, tuning_df = tune_svm_hyperparameters(X_train, y_train, X_val, y_val)
    best_val_row = tuning_df.sort_values(by="val_macro_f1", ascending=False).iloc[0]
    val_metrics = {
        "val_accuracy": best_val_row["val_accuracy"],
        "val_macro_f1": best_val_row["val_macro_f1"],
        "val_weighted_f1": best_val_row["val_weighted_f1"],
    }

    # 3. Fit Final Pipeline with Best Parameters on Train Data
    print(f"\n[*] Fitting final SVM baseline pipeline on full training data ({len(X_train):,} samples) ...")
    t0 = time.perf_counter()
    final_pipeline = build_svm_pipeline(
        C=best_params["C"],
        gamma=best_params["gamma"],
        class_weight=best_params["class_weight"],
        probability=False,
    )
    final_pipeline.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    print(f"[*] Model fitting completed in {fit_time:.2f}s")

    # 4. Save Model Pipeline
    model_path = output_dir / "svm_baseline.joblib"
    joblib.dump(final_pipeline, model_path)
    print(f"[*] Saved complete pipeline (StandardScaler + SVC) -> {model_path}")

    # 5. Evaluate on Untouched Test Set
    eval_results = evaluate_on_test_set(final_pipeline, X_test, y_test, CLASS_NAMES)

    # 6. Generate Visualizations
    print("\n[*] Generating evaluation visualizations ...")
    generated_plots = generate_svm_visualizations(
        eval_results, CLASS_NAMES, tuning_df, best_params, output_dir
    )
    for p in generated_plots:
        print(f"  * Generated plot -> {p}")

    # 7. Generate Audit Report
    report_path = output_dir / "svm_baseline_report.txt"
    generate_svm_report(
        train_size=len(X_train),
        val_size=len(X_val),
        test_size=len(X_test),
        feature_cols=feature_cols,
        tuning_df=tuning_df,
        best_params=best_params,
        val_metrics=val_metrics,
        eval_results=eval_results,
        report_path=report_path,
        model_save_path=model_path,
    )
    print(f"[*] Audit report written -> {report_path}")

    # 8. Print Executive Summary
    print("\n" + "=" * 80)
    print("  BASELINE SVM EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"  Best Hyperparameters : C={best_params['C']}, gamma={best_params['gamma']}, class_weight={best_params['class_weight']}")
    print(f"  Validation Accuracy  : {val_metrics['val_accuracy']*100:.2f}%")
    print(f"  Validation Macro F1  : {val_metrics['val_macro_f1']:.4f}")
    print(f"  Test Accuracy        : {eval_results['accuracy']*100:.2f}%")
    print(f"  Test Macro F1-Score  : {eval_results['macro_f1']:.4f}")
    print(f"  Test Weighted F1-Score: {eval_results['weighted_f1']:.4f}")
    print("\n  Per-Class Test Results:")
    print(eval_results["per_class_metrics"].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"  Test Inference Speed : {eval_results['ms_per_sample']:.2f} ms / sample ({eval_results['total_inference_time_s']:.3f}s total)")
    print(f"  Model Saved At       : {model_path}")
    print(f"  Report Saved At      : {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
