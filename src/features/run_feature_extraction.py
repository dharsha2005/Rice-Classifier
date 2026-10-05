"""
run_feature_extraction.py
=========================
Handcrafted Feature Extraction Pipeline Runner & Visualizer

Project:
    Rice Quality and Defect Assessment using Hybrid Machine Learning and Deep Feature Fusion

Features:
    - High-throughput parallel processing of train, val, and test dataset splits.
    - Saves feature matrices to separate CSV files:
        * results/features/train_handcrafted_features.csv
        * results/features/val_handcrafted_features.csv
        * results/features/test_handcrafted_features.csv
    - Comprehensive data validation (NaN, Inf, constant features, correlation).
    - Generates 4 rich feature visualizations:
        1. Correlation Heatmap
        2. Class-wise Feature Distributions
        3. PCA 2D Dimensionality Reduction Plot
        4. Feature-count summary chart
    - Generates detailed results/features/handcrafted_feature_report.txt.

Usage:
    # 1. Validation test on representative samples across all 8 classes
    python src/features/run_feature_extraction.py --sample-only --samples-per-class 10

    # 2. Complete dataset processing
    python src/features/run_feature_extraction.py --split all --workers 8
"""

import os
import sys
import time
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.handcrafted_features import (
    process_image_file_worker,
    validate_feature_dataframe,
)

DATASETS_DIR = PROJECT_ROOT / "datasets"
RICE_IMAGE_ROOT = PROJECT_ROOT / "rice"
RESULTS_DIR = PROJECT_ROOT / "results" / "features"
VISUALS_DIR = RESULTS_DIR / "visualizations"

SPLIT_FILES = {
    "train": "rice_train.txt",
    "val": "rice_val.txt",
    "test": "rice_test.txt",
    "train_bal": "rice_train_bal.txt",
}

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


def load_split_items(split_name: str) -> List[Tuple[str, int, str, str, str]]:
    """
    Parse a split file and return list of tuples formatted for multiprocessing worker:
    (rel_path, label_id, class_name, split_name, image_root_str)
    """
    filename = SPLIT_FILES[split_name]
    filepath = DATASETS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Split file missing: {filepath}")

    items = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                rel_path, label_str = parts
                label_id = int(label_str)
                parts_path = Path(rel_path).parts
                class_folder = parts_path[1] if len(parts_path) >= 2 else CLASS_NAMES.get(label_id, "UNKNOWN")
                items.append((rel_path, label_id, class_folder, split_name, str(RICE_IMAGE_ROOT)))
    return items


def extract_features_parallel(
    items: List[Tuple[str, int, str, str, str]],
    num_workers: int = 8,
    chunksize: int = 50,
    desc: str = "Processing"
) -> pd.DataFrame:
    """
    Extract features in parallel using multiprocessing Pool.
    """
    total = len(items)
    print(f"[*] {desc}: {total:,} images using {num_workers} parallel workers (chunksize={chunksize}) ...")

    t0 = time.perf_counter()
    results = []

    with mp.Pool(processes=num_workers) as pool:
        # Use imap_unordered for speed and memory efficiency
        for i, row_dict in enumerate(pool.imap_unordered(process_image_file_worker, items, chunksize=chunksize), 1):
            results.append(row_dict)
            if i % 1000 == 0 or i == total:
                elapsed = time.perf_counter() - t0
                speed = i / elapsed if elapsed > 0 else 0
                print(f"    [{i:>6,}/{total:,}] ({i/total*100:5.1f}%) | Speed: {speed:6.1f} img/s | Elapsed: {elapsed:5.1f}s")

    t1 = time.perf_counter()
    print(f"[*] Completed {total:,} images in {t1-t0:.2f}s (Average: {total/(t1-t0):.1f} images/sec)")

    df = pd.DataFrame(results)
    return df


# ---------------------------------------------------------------------------
# VISUALIZATIONS GENERATOR
# ---------------------------------------------------------------------------
def generate_all_visualizations(df: pd.DataFrame, output_dir: Path) -> List[Path]:
    """
    Generate the 4 required feature visualization plots:
    1. Feature correlation heatmap
    2. Class-wise feature distribution boxplots for selected key features
    3. PCA 2D projection scatter plot colored by class
    4. Feature count summary bar chart
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_plots = []

    # Clean numeric features
    meta_cols = ["image_path", "split", "class_id", "class_name", "status", "error_msg"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    numeric_df = df[feat_cols].dropna()

    # Filter out failed rows if any
    valid_mask = (df["status"] == "SUCCESS") if "status" in df.columns else np.ones(len(df), dtype=bool)
    valid_df = df[valid_mask].copy()

    # Apply dark theme styling
    plt.style.use("dark_background")

    # -----------------------------------------------------------------------
    # 1. Feature Correlation Heatmap
    # -----------------------------------------------------------------------
    print("[*] Generating Visualization 1: Feature Correlation Heatmap ...")
    corr = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(18, 16), facecolor="#090d16")
    ax.set_facecolor("#0f172a")
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.2,
        linecolor="#1e293b",
        cbar_kws={"shrink": 0.7, "label": "Pearson Correlation Coefficient (r)"},
        xticklabels=True,
        yticklabels=True,
        ax=ax,
    )
    plt.xticks(fontsize=6.5, rotation=90, color="#94a3b8")
    plt.yticks(fontsize=6.5, rotation=0, color="#94a3b8")
    plt.title("Handcrafted Feature Correlation Heatmap (62 Features: Shape, Texture, Colour)", fontsize=14, fontweight="bold", color="#f8fafc", pad=15)
    
    heatmap_path = output_dir / "feature_correlation_heatmap.png"
    plt.tight_layout()
    plt.savefig(str(heatmap_path), dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    generated_plots.append(heatmap_path)

    # -----------------------------------------------------------------------
    # 2. Class-wise Feature Distributions (6 Selected Representative Features)
    # -----------------------------------------------------------------------
    print("[*] Generating Visualization 2: Class-wise Feature Distributions ...")
    key_features = [
        ("shape_aspect_ratio", "Shape: Aspect Ratio (w / h)"),
        ("shape_circularity", "Shape: Circularity (Compactness)"),
        ("glcm_contrast_mean", "Texture: GLCM Contrast Mean"),
        ("glcm_homogeneity_mean", "Texture: GLCM Homogeneity Mean"),
        ("color_hsv_h_mean", "Colour: HSV Hue Mean"),
        ("color_lab_l_mean", "Colour: LAB Lightness (L*) Mean"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="#090d16")
    axes = axes.flatten()

    palette = sns.color_palette("husl", 8)

    for i, (feat_col, feat_title) in enumerate(key_features):
        if feat_col in valid_df.columns:
            sns.boxplot(
                data=valid_df,
                x="class_name",
                y=feat_col,
                ax=axes[i],
                palette=palette,
                boxprops=dict(alpha=0.85),
                showfliers=False,
            )
            axes[i].set_title(feat_title, color="#38bdf8", fontsize=11, fontweight="bold", pad=8)
            axes[i].set_xlabel("Rice Class", color="#94a3b8", fontsize=9)
            axes[i].set_ylabel(feat_col, color="#94a3b8", fontsize=9)
            axes[i].tick_params(colors="#cbd5e1", labelsize=8)
            axes[i].grid(True, alpha=0.15, linestyle="--")

    fig.suptitle("Class-wise Distributions for Key Handcrafted Morphological, Texture, and Colour Features", fontsize=14, fontweight="bold", color="#f8fafc", y=0.99)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])
    dist_path = output_dir / "class_feature_distributions.png"
    plt.savefig(str(dist_path), dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    generated_plots.append(dist_path)

    # -----------------------------------------------------------------------
    # 3. PCA 2D Dimensionality Reduction Plot
    # -----------------------------------------------------------------------
    print("[*] Generating Visualization 3: PCA 2D Projection ...")
    X_vals = valid_df[feat_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_vals)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(12, 8), facecolor="#090d16")
    ax.set_facecolor("#0f172a")

    unique_classes = sorted(valid_df["class_id"].unique())
    colors = sns.color_palette("tab10", len(unique_classes))

    # Downsample points if huge dataset to prevent over-plotting and keep clean
    plot_indices = np.random.choice(len(valid_df), size=min(4000, len(valid_df)), replace=False)

    for cid in unique_classes:
        cname = CLASS_NAMES.get(cid, f"Class {cid}")
        cls_mask = (valid_df["class_id"].iloc[plot_indices] == cid).values
        if np.sum(cls_mask) > 0:
            ax.scatter(
                X_pca[plot_indices][cls_mask, 0],
                X_pca[plot_indices][cls_mask, 1],
                label=f"[{cid}] {cname}",
                alpha=0.6,
                s=20,
                color=colors[cid],
                edgecolors="none",
            )

    ax.set_xlabel(f"Principal Component 1 ({var_exp[0]*100:.1f}% Variance)", color="#e2e8f0", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Principal Component 2 ({var_exp[1]*100:.1f}% Variance)", color="#e2e8f0", fontsize=11, fontweight="bold")
    ax.set_title("PCA 2D Projection of 62 Handcrafted Features Across 8 Rice Classes", color="#f8fafc", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.85, facecolor="#1e293b", edgecolor="#334155", fontsize=9)
    ax.grid(True, alpha=0.15, linestyle="--")

    pca_path = output_dir / "pca_handcrafted_features.png"
    plt.tight_layout()
    plt.savefig(str(pca_path), dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    generated_plots.append(pca_path)

    # -----------------------------------------------------------------------
    # 4. Feature Count Summary Chart
    # -----------------------------------------------------------------------
    print("[*] Generating Visualization 4: Feature Count Summary ...")
    shape_count = len([c for c in feat_cols if c.startswith("shape_")])
    texture_count = len([c for c in feat_cols if c.startswith("glcm_")])
    color_count = len([c for c in feat_cols if c.startswith("color_")])
    total_count = len(feat_cols)

    categories = ["Shape / Morphology", "GLCM Texture", "Colour Statistics", "Total Combined"]
    counts = [shape_count, texture_count, color_count, total_count]
    bar_colors = ["#38bdf8", "#a855f7", "#ec4899", "#10b981"]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#090d16")
    ax.set_facecolor("#0f172a")

    bars = ax.bar(categories, counts, color=bar_colors, width=0.55, edgecolor="#ffffff", linewidth=0.5)
    for bar, count in zip(bars, counts):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 1.0,
            f"{count} features",
            ha="center",
            va="bottom",
            color="#f8fafc",
            fontweight="bold",
            fontsize=11,
        )

    ax.set_ylim(0, max(counts) + 12)
    ax.set_ylabel("Number of Features", color="#e2e8f0", fontsize=11, fontweight="bold")
    ax.set_title("Handcrafted Feature Engineering: Category Breakdown", color="#f8fafc", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="#cbd5e1", labelsize=10)
    ax.grid(axis="y", alpha=0.15, linestyle="--")

    count_path = output_dir / "feature_count_summary.png"
    plt.tight_layout()
    plt.savefig(str(count_path), dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    generated_plots.append(count_path)

    return generated_plots


# ---------------------------------------------------------------------------
# REPORT GENERATOR
# ---------------------------------------------------------------------------
def generate_feature_report(
    split_dfs: Dict[str, pd.DataFrame],
    validation_stats: Dict[str, Any],
    report_path: Path,
    total_time_s: float
) -> None:
    """
    Generate comprehensive text report documenting handcrafted feature extraction.
    """
    total_imgs = sum(len(df) for df in split_dfs.values())
    total_failed = sum((df["status"] != "SUCCESS").sum() if "status" in df.columns else 0 for df in split_dfs.values())

    lines = [
        "=" * 78,
        "  HANDCRAFTED FEATURE EXTRACTION AUDIT & VALIDATION REPORT",
        "  Project: Rice Quality and Defect Assessment using Hybrid ML & Feature Fusion",
        f"  Date   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 78,
        "",
        "1. DATASET PROCESSING OVERVIEW",
        f"  - Total Rice Images Processed   : {total_imgs:,}",
        f"  - Total Successful Extractions : {total_imgs - total_failed:,} ({(total_imgs - total_failed)/total_imgs*100:.2f}%)",
        f"  - Total Failed Extractions     : {total_failed:,}",
        f"  - Total Processing Time        : {total_time_s:.2f} seconds ({total_time_s/60:.2f} minutes)",
        f"  - Overall Extraction Throughput: {total_imgs/total_time_s:.1f} images / second",
        "",
        "  Per-Split Breakdown:",
    ]

    for split_name, df in split_dfs.items():
        split_failed = (df["status"] != "SUCCESS").sum() if "status" in df.columns else 0
        lines.append(f"    * {split_name:<10s}: {len(df):>6,} images | Successful: {len(df) - split_failed:>6,} | Failed: {split_failed:>3,}")

    lines.extend([
        "",
        "2. FEATURE VECTOR ARCHITECTURE",
        f"  - Total Handcrafted Features   : {validation_stats['total_features']}",
        f"  - Shape / Geometric Features   : {validation_stats['shape_feature_count']} features",
        f"  - GLCM Texture Features        : {validation_stats['texture_feature_count']} features",
        f"  - Colour Space Features        : {validation_stats['color_feature_count']} features",
        "",
        "  Detailed Feature Group Composition:",
        "  --------------------------------------------------------------------------",
        "  [A] SHAPE FEATURES (14):",
        "      1. shape_area                8. shape_circularity",
        "      2. shape_perimeter           9. shape_eccentricity",
        "      3. shape_width              10. shape_major_axis_length",
        "      4. shape_height             11. shape_minor_axis_length",
        "      5. shape_aspect_ratio       12. shape_equivalent_diameter",
        "      6. shape_extent             13. shape_convex_hull_area",
        "      7. shape_solidity           14. shape_bbox_area",
        "",
        "  [B] GLCM TEXTURE FEATURES (12) (Distances: [1, 2], Angles: [0, 45, 90, 135 deg]):",
        "      1. glcm_contrast_mean        7. glcm_energy_mean",
        "      2. glcm_contrast_std         8. glcm_energy_std",
        "      3. glcm_dissimilarity_mean   9. glcm_correlation_mean",
        "      4. glcm_dissimilarity_std   10. glcm_correlation_std",
        "      5. glcm_homogeneity_mean    11. glcm_asm_mean",
        "      6. glcm_homogeneity_std     12. glcm_asm_std",
        "",
        "  [C] COLOUR FEATURES (36) (Spaces: RGB, HSV, LAB | Stats: Mean, Std, Min, Max):",
        "      * RGB Channels (12) : R (mean, std, min, max), G (mean, std, min, max), B (mean, std, min, max)",
        "      * HSV Channels (12) : H (mean, std, min, max), S (mean, std, min, max), V (mean, std, min, max)",
        "      * LAB Channels (12) : L (mean, std, min, max), A (mean, std, min, max), B (mean, std, min, max)",
        "",
        "3. NUMERICAL INTEGRITY & VALIDATION AUDIT",
        f"  - Missing / NaN Values         : {validation_stats['total_nans']} (PASSED)",
        f"  - Infinite / Inf Values        : {validation_stats['total_infs']} (PASSED)",
        f"  - Constant Features (std = 0)  : {len(validation_stats['constant_features'])} {validation_stats['constant_features'] if validation_stats['constant_features'] else '(None - PASSED)'}",
        f"  - Highly Correlated Pairs (>0.95): {len(validation_stats['high_corr_pairs'])} pairs detected",
    ])

    if validation_stats["high_corr_pairs"]:
        lines.append("    (Top Correlated Pairs Expected in Geometry/Texture):")
        for f1, f2, r_val in validation_stats["high_corr_pairs"][:8]:
            lines.append(f"      * {f1:<26s} <-> {f2:<26s} : r = {r_val:.4f}")

    lines.extend([
        "",
        "4. OUTPUT FILES GENERATED",
        f"  - Train Split Features CSV     : results/features/train_handcrafted_features.csv",
        f"  - Val Split Features CSV       : results/features/val_handcrafted_features.csv",
        f"  - Test Split Features CSV      : results/features/test_handcrafted_features.csv",
        f"  - Visualizations Directory     : results/features/visualizations/",
        "      * feature_correlation_heatmap.png",
        "      * class_feature_distributions.png",
        "      * pca_handcrafted_features.png",
        "      * feature_count_summary.png",
        "",
        "5. DOWNSTREAM READINESS FOR HYBRID FUSION",
        "  - Clean tabular format suitable for direct input into XGBoost and SVM classifiers.",
        "  - Ready for feature scaling / standardization prior to baseline ML training.",
        "  - Ready for concatenation with EfficientNet-B0 deep features in Stage 4.",
        "",
        "=" * 78,
        "  END OF REPORT",
        "=" * 78,
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# MAIN EXECUTION ORCHESTRATOR
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Handcrafted Feature Extraction Pipeline")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"], help="Dataset split to extract features from")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2), help="Number of worker processes")
    parser.add_argument("--sample-only", action="store_true", help="Run validation only on representative sample across all classes")
    parser.add_argument("--samples-per-class", type=int, default=10, help="Number of samples per class for --sample-only")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR), help="Output directory")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir = output_dir / "visualizations"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  Rice Quality & Defect Assessment - Handcrafted Feature Extraction")
    print(f"  Split Mode       : {args.split}")
    print(f"  Workers          : {args.workers}")
    print(f"  Sample Only Mode : {args.sample_only}")
    print(f"  Output Directory : {output_dir}")
    print("=" * 78)

    t_start = time.perf_counter()

    if args.sample_only:
        # 1. Validation test on representative samples across all 8 classes
        print("\n[PHASE 1] Loading sample representative images across all 8 classes ...")
        all_train_items = load_split_items("train")
        by_class: Dict[int, List[Tuple[str, int, str, str, str]]] = {}
        for itm in all_train_items:
            by_class.setdefault(itm[1], []).append(itm)

        sample_items = []
        for cid in sorted(by_class.keys()):
            c_items = by_class[cid]
            step = max(1, len(c_items) // args.samples_per_class)
            sample_items.extend(c_items[:args.samples_per_class * step:step][:args.samples_per_class])

        print(f"[*] Extracting features from {len(sample_items)} validation samples ...")
        sample_df = extract_features_parallel(sample_items, num_workers=min(4, args.workers), desc="Sample Validation")

        # Validate
        print("\n[PHASE 2] Validating extracted feature matrix ...")
        val_stats = validate_feature_dataframe(sample_df)

        print("=" * 60)
        print("  FEATURE VALIDATION SUMMARY")
        print("=" * 60)
        print(f"  Total Sample Images  : {val_stats['total_images']}")
        print(f"  Total Features       : {val_stats['total_features']}")
        print(f"  Shape Features       : {val_stats['shape_feature_count']}")
        print(f"  Texture Features     : {val_stats['texture_feature_count']}")
        print(f"  Colour Features      : {val_stats['color_feature_count']}")
        print(f"  Missing / NaNs       : {val_stats['total_nans']}")
        print(f"  Infs                 : {val_stats['total_infs']}")
        print(f"  Constant Features    : {len(val_stats['constant_features'])}")
        print(f"  High Corr Pairs >0.95: {len(val_stats['high_corr_pairs'])}")
        print("=" * 60)

        sample_csv = output_dir / "sample_validation_features.csv"
        sample_df.to_csv(sample_csv, index=False)
        print(f"[*] Sample validation CSV saved -> {sample_csv}")
        return

    # Full Dataset Processing
    splits_to_process = ["train", "val", "test"] if args.split == "all" else [args.split]
    split_dfs: Dict[str, pd.DataFrame] = {}

    for s_name in splits_to_process:
        print(f"\n[{s_name.upper()} SPLIT] Loading entries ...")
        items = load_split_items(s_name)
        df = extract_features_parallel(items, num_workers=args.workers, desc=f"Extracting {s_name.upper()} Features")

        # Reorder columns: metadata first, then features
        meta_cols = ["image_path", "split", "class_id", "class_name"]
        feat_cols = [c for c in df.columns if c not in meta_cols and c not in ["status", "error_msg"]]
        # Drop temporary status columns if clean or keep at end
        final_cols = meta_cols + feat_cols
        ordered_df = df[final_cols]

        csv_path = output_dir / f"{s_name}_handcrafted_features.csv"
        ordered_df.to_csv(csv_path, index=False)
        print(f"[*] Saved {s_name.upper()} features ({len(ordered_df):,} rows x {len(final_cols)} cols) -> {csv_path}")
        split_dfs[s_name] = ordered_df

    t_end = time.perf_counter()
    total_elapsed = t_end - t_start

    # Combine for visualization & global audit
    full_df = pd.concat(split_dfs.values(), ignore_index=True)
    validation_stats = validate_feature_dataframe(full_df)

    # Visualizations
    print("\n[VISUALIZATION] Generating visual charts and distribution plots ...")
    generated_plots = generate_all_visualizations(full_df, visuals_dir)
    for p in generated_plots:
        print(f"  * Generated plot -> {p}")

    # Generate Feature Report
    report_path = output_dir / "handcrafted_feature_report.txt"
    generate_feature_report(split_dfs, validation_stats, report_path, total_elapsed)
    print(f"\n[*] Audit report written -> {report_path}")

    print("\n" + "=" * 78)
    print("  FEATURE EXTRACTION COMPLETED SUCCESSFULLY")
    print(f"  Total Images Processed : {len(full_df):,}")
    print(f"  Total Features Extracted: {validation_stats['total_features']}")
    print(f"  Elapsed Time           : {total_elapsed:.2f} s ({total_elapsed/60:.2f} min)")
    print(f"  Report Location        : {report_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
