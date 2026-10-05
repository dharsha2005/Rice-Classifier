"""Extract ImageNet EfficientNet-B0 features for fixed rice dataset splits."""

from __future__ import annotations

import argparse
import platform
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "datasets"
IMAGE_ROOT = PROJECT_ROOT / "rice"
OUTPUT_DIR = PROJECT_ROOT / "results" / "features" / "deep"
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"
SPLITS = {"train": "rice_train.txt", "val": "rice_val.txt", "test": "rice_test.txt"}
CLASS_NAMES = {0: "0_NOR", 1: "1_F&S", 2: "2_SD", 3: "3_MY", 4: "4_AP", 5: "5_BN", 6: "6_UN", 7: "7_IM"}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.efficientnet_features import ImageEntry, extract_split_features, load_efficientnet_b0


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_manifest(split: str, limit: int | None = None) -> List[ImageEntry]:
    path = DATASETS_DIR / SPLITS[split]
    entries: List[ImageEntry] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if not value:
                continue
            relative_path, label_text = value.rsplit(" ", 1)
            class_id = int(label_text)
            class_name = Path(relative_path).parts[1]
            entries.append(ImageEntry(relative_path, split, class_id, class_name))
            if limit is not None and len(entries) >= limit:
                break
    return entries


def save_feature_csv(entries: Sequence[ImageEntry], features: np.ndarray, path: Path) -> None:
    if len(entries) != len(features):
        raise ValueError(f"Metadata rows ({len(entries)}) do not match feature rows ({len(features)})")
    if features.ndim != 2 or features.shape[1] == 0:
        raise ValueError(f"Expected a 2D non-empty feature matrix, got {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("Cannot save feature matrix containing NaN or infinite values")
    metadata = pd.DataFrame([entry.__dict__ for entry in entries])
    feature_names = [f"deep_feature_{index:03d}" for index in range(1, features.shape[1] + 1)]
    frame = pd.concat([metadata, pd.DataFrame(features, columns=feature_names)], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def validate_outputs(frames: Dict[str, pd.DataFrame], expected_counts: Dict[str, int], feature_dimension: int) -> Tuple[int, int]:
    nan_count = 0
    infinite_count = 0
    metadata = ["image_path", "split", "class_id", "class_name"]
    for split, frame in frames.items():
        feature_columns = [column for column in frame.columns if column.startswith("deep_feature_")]
        if len(frame) != expected_counts[split] or len(feature_columns) != feature_dimension:
            raise ValueError(f"Invalid {split} output shape: {frame.shape}, expected rows={expected_counts[split]}, features={feature_dimension}")
        if frame[metadata].isna().any().any():
            raise ValueError(f"Missing metadata values in {split} output")
        values = frame[feature_columns].to_numpy(dtype=np.float64)
        nan_count += int(np.isnan(values).sum())
        infinite_count += int(np.isinf(values).sum())
        if nan_count or infinite_count:
            raise ValueError(f"Invalid feature values in {split} output")
    return nan_count, infinite_count


def create_visualizations(frames: Dict[str, pd.DataFrame], feature_dimension: int) -> Tuple[float, str]:
    combined = pd.concat(frames.values(), ignore_index=True)
    columns = [f"deep_feature_{index:03d}" for index in range(1, feature_dimension + 1)]
    scaled = StandardScaler().fit_transform(combined[columns].to_numpy(dtype=np.float32))
    pca = PCA(n_components=2, random_state=42)
    projected = pca.fit_transform(scaled)
    plot_frame = pd.DataFrame({"pc1": projected[:, 0], "pc2": projected[:, 1], "class_name": combined["class_name"], "split": combined["split"]})
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.scatterplot(data=plot_frame, x="pc1", y="pc2", hue="class_name", style="split", s=16, alpha=0.65, ax=ax)
    ax.set_title("EfficientNet-B0 Deep Features: PCA by Class")
    fig.tight_layout()
    fig.savefig(VISUALIZATION_DIR / "pca_deep_features.png", dpi=180)
    plt.close(fig)

    class_centers = plot_frame.groupby("class_name", sort=False)[["pc1", "pc2"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(data=class_centers, x="pc1", y="pc2", hue="class_name", s=180, ax=ax)
    for _, row in class_centers.iterrows():
        ax.annotate(row["class_name"], (row["pc1"], row["pc2"]), xytext=(5, 5), textcoords="offset points")
    ax.set_title("EfficientNet-B0 PCA Class-Wise Cluster Centers")
    fig.tight_layout()
    fig.savefig(VISUALIZATION_DIR / "class_wise_clustering.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(["PC1", "PC2"], pca.explained_variance_ratio_ * 100.0, color=["#38bdf8", "#10b981"])
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("PCA Explained Variance")
    fig.tight_layout()
    fig.savefig(VISUALIZATION_DIR / "pca_explained_variance.png", dpi=180)
    plt.close(fig)
    observation = f"The first two PCA components explain {pca.explained_variance_ratio_.sum() * 100:.2f}% of standardized feature variance ({pca.explained_variance_ratio_[0] * 100:.2f}% + {pca.explained_variance_ratio_[1] * 100:.2f}%)."
    return float(pca.explained_variance_ratio_.sum()), observation


def write_report(
    path: Path, model_name: str, weights: str, device: torch.device, batch_size: int,
    counts: Dict[str, int], feature_dimension: int, elapsed: float, failed_images: List[Tuple[ImageEntry, str]],
    nan_count: int, infinite_count: int, pca_observation: str,
) -> None:
    total = sum(counts.values())
    lines = [
        "EFFICIENTNET-B0 DEEP FEATURE EXTRACTION REPORT", "",
        "Research role: deep feature representation only; no classifier was trained.",
        "This output is not presented as the proposed/final model.", "",
        f"Model name: {model_name}", f"Pretrained weights: {weights}", "Input size: 224 x 224 x 3",
        f"Feature dimension: {feature_dimension}", f"Device: {device}", f"Batch size: {batch_size}",
        f"Python version: {platform.python_version()}", f"PyTorch version: {torch.__version__}",
        f"Torchvision version: {__import__('torchvision').__version__}", "",
        f"Train image count: {counts['train']}", f"Validation image count: {counts['val']}", f"Test image count: {counts['test']}",
        f"Total extraction time: {elapsed:.3f} seconds", f"Images/second: {total / elapsed:.3f}",
        f"Average time/image: {elapsed / total:.6f} seconds", f"Failed images: {len(failed_images)}",
        f"NaN count: {nan_count}", f"Infinite value count: {infinite_count}", "", "PCA observations:", pca_observation, "",
        "Failed image details:",
    ]
    lines.extend(f"- {entry.image_path}: {reason}" for entry, reason in failed_images)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract EfficientNet-B0 deep features")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-images", type=int, default=None, help="Limit each split for a smoke test")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    set_seed()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, feature_dimension, weights = load_efficientnet_b0(device)
    print(model)
    print(f"Feature vector dimension: {feature_dimension}")
    all_failed: List[Tuple[ImageEntry, str]] = []
    frames: Dict[str, pd.DataFrame] = {}
    counts: Dict[str, int] = {}
    total_elapsed = 0.0
    for split in SPLITS:
        entries = load_manifest(split, args.max_images)
        counts[split] = len(entries)
        print(f"Extracting {split}: {len(entries):,} images")
        result = extract_split_features(entries, IMAGE_ROOT, model, device, args.batch_size)
        total_elapsed += result.elapsed_seconds
        all_failed.extend(result.failed_images)
        if result.failed_images:
            for entry, reason in result.failed_images:
                print(f"FAILED: {entry.image_path} ({reason})")
            raise RuntimeError(f"{len(result.failed_images)} {split} images failed; no incomplete CSV was written")
        output_path = output_dir / f"{split}_efficientnet_features.csv"
        save_feature_csv(result.entries, result.features, output_path)
        frames[split] = pd.read_csv(output_path)
        print(f"Saved {output_path}")
    nan_count, infinite_count = validate_outputs(frames, counts, feature_dimension)
    _, pca_observation = create_visualizations(frames, feature_dimension)
    report_path = output_dir / "efficientnet_feature_report.txt"
    write_report(report_path, "EfficientNet-B0", weights, device, args.batch_size, counts, feature_dimension, total_elapsed, all_failed, nan_count, infinite_count, pca_observation)
    total = sum(counts.values())
    print(f"Images processed: {total:,}")
    print(f"Failed images: {len(all_failed)}")
    print(f"NaN count: {nan_count}; infinite count: {infinite_count}")
    print(f"Extraction time: {total_elapsed:.3f}s; images/second: {total / total_elapsed:.3f}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
