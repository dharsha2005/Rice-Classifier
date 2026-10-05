"""
preprocess.py
=============
Rice Quality and Defect Assessment - Image Preprocessing & Grain Segmentation Pipeline

Project:
    Rice Quality and Defect Assessment using Hybrid Machine Learning and Deep Feature Fusion

Purpose:
    - Load raw rice grain images from GrainSet split files.
    - Preserve original class labels without re-encoding or rebalancing.
    - Denoise raw images without destroying sharp grain boundaries.
    - Convert images into multiple color spaces (RGB, Grayscale, HSV, LAB).
    - Segment the rice grain from the background using Otsu's thresholding & morphological refinement.
    - Extract grain contour and compute morphological properties (area, perimeter, bounding box, aspect ratio).
    - Create clean binary masks and extract segmented grain foreground.
    - Standardize and center-crop/pad segmented grains into a consistent 224x224 resolution for EfficientNet-B0.
    - Generate 4-panel visual comparisons (Original -> Grayscale -> Mask -> Segmented/Cropped Grain).
    - Generate a comprehensive preprocessing report.

Usage:
    python src/preprocessing/preprocess.py --samples 20 --split train
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI environments
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Project Paths & Defaults
# ---------------------------------------------------------------------------
PROJECT_ROOT     = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR     = PROJECT_ROOT / "datasets"
RICE_IMAGE_ROOT  = PROJECT_ROOT / "rice"
RESULTS_DIR      = PROJECT_ROOT / "results" / "preprocessing"
REPORTS_DIR      = PROJECT_ROOT / "reports"

SPLIT_FILES = {
    "train":     "rice_train.txt",
    "val":       "rice_val.txt",
    "test":      "rice_test.txt",
    "train_bal": "rice_train_bal.txt",
}

# Standard EfficientNet-B0 input dimensions
DEFAULT_TARGET_SIZE = (224, 224)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class GrainMorphology:
    """Morphological properties extracted from the segmented grain contour."""
    area: float
    perimeter: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    aspect_ratio: float
    extent: float
    solidity: float
    centroid: Tuple[float, float]    # (cx, cy)
    min_area_rect: Tuple[Tuple[float, float], Tuple[float, float], float]  # center, size, angle


@dataclass
class PreprocessingResult:
    """Encapsulates all intermediate and final outputs for a single processed image."""
    image_name: str
    relative_path: str
    label_id: int
    class_name: str
    original_shape: Tuple[int, int, int]  # (H, W, C)
    processed_shape: Tuple[int, int, int]  # (H, W, C)
    
    # Image representations
    original_bgr: np.ndarray
    rgb: np.ndarray
    gray: np.ndarray
    hsv: np.ndarray
    lab: np.ndarray
    gray_denoised: np.ndarray
    
    # Segmentation & Masking
    binary_mask: np.ndarray
    contour: Optional[np.ndarray]
    morphology: Optional[GrainMorphology]
    
    # Output images
    segmented_full: np.ndarray       # Full size with background zeroed out
    segmented_cropped: np.ndarray    # Tight bounding crop with margin
    standardized_grain: np.ndarray   # Standardized (224, 224, 3) padded/centered
    
    # Execution metadata
    success: bool = True
    error_message: Optional[str] = None
    processing_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Rice Preprocessor Pipeline Class
# ---------------------------------------------------------------------------
class RicePreprocessor:
    """
    Modular, robust preprocessing and segmentation pipeline for rice grain images.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
        blur_ksize: Tuple[int, int] = (5, 5),
        morph_ksize: Tuple[int, int] = (5, 5),
        min_grain_area: float = 500.0,
        margin_ratio: float = 0.05,
        fill_scale: float = 0.88,
    ):
        """
        Initialize preprocessing pipeline parameters.

        Args:
            target_size: Final output image dimensions (width, height), default (224, 224).
            blur_ksize: Gaussian blur kernel size for edge-preserving noise reduction.
            morph_ksize: Morphological structuring element size for hole closing / noise filtering.
            min_grain_area: Minimum contour pixel area to be considered a valid rice grain.
            margin_ratio: Margin to add around bounding box when cropping grain (0.05 = 5%).
            fill_scale: Target scale to fit grain inside the standardized output square.
        """
        self.target_size = target_size
        self.blur_ksize = blur_ksize
        self.morph_ksize = morph_ksize
        self.min_grain_area = min_grain_area
        self.margin_ratio = margin_ratio
        self.fill_scale = fill_scale
        
        # Precompute structuring elements
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.morph_ksize)

    def load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Safely read an image from disk in BGR format.
        """
        if not image_path.exists():
            return None
        # Use cv2.imdecode to handle any path encoding issues gracefully on Windows
        try:
            img_bytes = np.fromfile(str(image_path), dtype=np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    def convert_color_spaces(self, bgr: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Convert BGR image to RGB, Grayscale, HSV, and LAB colour spaces.
        """
        return {
            "bgr": bgr,
            "rgb": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            "gray": cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
            "hsv": cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV),
            "lab": cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB),
        }

    def denoise(self, gray: np.ndarray) -> np.ndarray:
        """
        Remove subtle sensor noise while preserving sharp rice-grain boundaries.
        Uses Gaussian Blur with calibrated kernel.
        """
        return cv2.GaussianBlur(gray, self.blur_ksize, 0)

    def segment_grain(
        self,
        gray_denoised: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[GrainMorphology], float]:
        """
        Segment the rice grain from background using Otsu's thresholding and morphological operations.

        Returns:
            clean_mask: Binary mask (uint8, 0 for background, 255 for rice grain).
            largest_contour: OpenCV contour of the primary rice grain.
            morphology: Morphological measurements (area, perimeter, bbox, aspect ratio, etc.).
            otsu_threshold: Computed Otsu threshold value.
        """
        # Step 1: Automatic Otsu Thresholding
        otsu_thresh, binary = cv2.threshold(
            gray_denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Step 2: Morphological Close (bridge small fissures/chalkiness) & Open (remove dust)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.morph_kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, self.morph_kernel)

        # Step 3: Find external contours
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return opened, None, None, otsu_thresh

        # Step 4: Identify largest contour (primary rice grain)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area < self.min_grain_area:
            return opened, None, None, otsu_thresh

        # Step 5: Construct clean mask of the single primary rice grain
        clean_mask = np.zeros_like(gray_denoised, dtype=np.uint8)
        cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # Step 6: Extract contour morphological properties
        perimeter = cv2.arcLength(largest_contour, True)
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / float(h) if h > 0 else 0.0
        bbox_area = float(w * h)
        extent = float(area) / bbox_area if bbox_area > 0 else 0.0

        # Convex hull and solidity
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0.0

        # Centroid moments
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
        else:
            cx, cy = float(x + w / 2), float(y + h / 2)

        # Minimum area rotated rectangle
        min_rect = cv2.minAreaRect(largest_contour)

        morphology = GrainMorphology(
            area=area,
            perimeter=perimeter,
            bbox=(x, y, w, h),
            aspect_ratio=aspect_ratio,
            extent=extent,
            solidity=solidity,
            centroid=(cx, cy),
            min_area_rect=min_rect,
        )

        return clean_mask, largest_contour, morphology, otsu_thresh

    def standardize_grain_image(
        self,
        bgr: np.ndarray,
        mask: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate:
        1. segmented_full: Full image with background zeroed out (black).
        2. segmented_cropped: Tightly cropped bounding box with small margin.
        3. standardized_grain: Centered, aspect-ratio preserved (224, 224, 3) image.
        """
        h_orig, w_orig = bgr.shape[:2]
        
        # 1. Full segmented image (masked background)
        segmented_full = cv2.bitwise_and(bgr, bgr, mask=mask)

        # 2. Crop with margin
        x, y, bw, bh = bbox
        margin_x = int(bw * self.margin_ratio)
        margin_y = int(bh * self.margin_ratio)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w_orig, x + bw + margin_x)
        y2 = min(h_orig, y + bh + margin_y)

        segmented_cropped = segmented_full[y1:y2, x1:x2]
        crop_h, crop_w = segmented_cropped.shape[:2]

        # 3. Standardize to target_size (224x224) maintaining aspect ratio
        target_w, target_h = self.target_size
        scale = min(target_w / crop_w, target_h / crop_h) * self.fill_scale
        nw, nh = max(1, int(crop_w * scale)), max(1, int(crop_h * scale))

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized_crop = cv2.resize(segmented_cropped, (nw, nh), interpolation=interp)

        standardized_grain = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        ox = (target_w - nw) // 2
        oy = (target_h - nh) // 2
        standardized_grain[oy:oy+nh, ox:ox+nw] = resized_crop

        return segmented_full, segmented_cropped, standardized_grain

    def process_bgr(
        self,
        bgr: np.ndarray,
        *,
        image_name: str = "inference",
        label_id: int = -1,
        class_name: str = "UNKNOWN",
        relative_path: str = "",
    ) -> PreprocessingResult:
        """Execute the full pipeline on an in-memory BGR image (upload/webcam parity with disk loads)."""
        import time
        start_time = time.perf_counter()

        if bgr is None or bgr.size == 0:
            return PreprocessingResult(
                image_name=image_name,
                relative_path=relative_path,
                label_id=label_id,
                class_name=class_name,
                original_shape=(0, 0, 0),
                processed_shape=self.target_size + (3,),
                original_bgr=np.zeros((1, 1, 3), dtype=np.uint8),
                rgb=np.zeros((1, 1, 3), dtype=np.uint8),
                gray=np.zeros((1, 1), dtype=np.uint8),
                hsv=np.zeros((1, 1, 3), dtype=np.uint8),
                lab=np.zeros((1, 1, 3), dtype=np.uint8),
                gray_denoised=np.zeros((1, 1), dtype=np.uint8),
                binary_mask=np.zeros((1, 1), dtype=np.uint8),
                contour=None,
                morphology=None,
                segmented_full=np.zeros((1, 1, 3), dtype=np.uint8),
                segmented_cropped=np.zeros((1, 1, 3), dtype=np.uint8),
                standardized_grain=np.zeros(self.target_size + (3,), dtype=np.uint8),
                success=False,
                error_message="Input image is empty.",
                processing_time_ms=0.0,
            )

        if bgr.dtype != np.uint8:
            bgr = bgr.astype(np.uint8)

        orig_shape = bgr.shape  # (H, W, C)

        # 1. Color space conversions
        color_spaces = self.convert_color_spaces(bgr)
        rgb = color_spaces["rgb"]
        gray = color_spaces["gray"]
        hsv = color_spaces["hsv"]
        lab = color_spaces["lab"]

        # 2. Denoising
        gray_denoised = self.denoise(gray)

        # 3. Grain Segmentation
        mask, contour, morphology, _ = self.segment_grain(gray_denoised)

        if contour is None or morphology is None:
            # Fallback if no valid grain contour was found
            segmented_full = np.zeros_like(bgr)
            segmented_cropped = np.zeros((10, 10, 3), dtype=np.uint8)
            standardized = np.zeros(self.target_size + (3,), dtype=np.uint8)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return PreprocessingResult(
                image_name=image_name,
                relative_path=relative_path,
                label_id=label_id,
                class_name=class_name,
                original_shape=orig_shape,
                processed_shape=self.target_size + (3,),
                original_bgr=bgr,
                rgb=rgb,
                gray=gray,
                hsv=hsv,
                lab=lab,
                gray_denoised=gray_denoised,
                binary_mask=mask,
                contour=None,
                morphology=None,
                segmented_full=segmented_full,
                segmented_cropped=segmented_cropped,
                standardized_grain=standardized,
                success=False,
                error_message="Segmentation failed: no valid grain contour detected",
                processing_time_ms=elapsed_ms
            )

        # 4. Standardize, Crop & Mask
        segmented_full, segmented_cropped, standardized_grain = self.standardize_grain_image(
            bgr, mask, morphology.bbox
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return PreprocessingResult(
            image_name=image_name,
            relative_path=relative_path,
            label_id=label_id,
            class_name=class_name,
            original_shape=orig_shape,
            processed_shape=standardized_grain.shape,
            original_bgr=bgr,
            rgb=rgb,
            gray=gray,
            hsv=hsv,
            lab=lab,
            gray_denoised=gray_denoised,
            binary_mask=mask,
            contour=contour,
            morphology=morphology,
            segmented_full=segmented_full,
            segmented_cropped=segmented_cropped,
            standardized_grain=standardized_grain,
            success=True,
            error_message=None,
            processing_time_ms=elapsed_ms
        )

    def process_image(
        self,
        image_path: Path,
        label_id: int = -1,
        class_name: str = "UNKNOWN",
        relative_path: str = ""
    ) -> PreprocessingResult:
        """
        Execute full preprocessing and segmentation pipeline on a single image.
        """
        image_name = image_path.name
        bgr = self.load_image(image_path)
        if bgr is None:
            return PreprocessingResult(
                image_name=image_name,
                relative_path=relative_path,
                label_id=label_id,
                class_name=class_name,
                original_shape=(0, 0, 0),
                processed_shape=self.target_size + (3,),
                original_bgr=np.zeros((1, 1, 3), dtype=np.uint8),
                rgb=np.zeros((1, 1, 3), dtype=np.uint8),
                gray=np.zeros((1, 1), dtype=np.uint8),
                hsv=np.zeros((1, 1, 3), dtype=np.uint8),
                lab=np.zeros((1, 1, 3), dtype=np.uint8),
                gray_denoised=np.zeros((1, 1), dtype=np.uint8),
                binary_mask=np.zeros((1, 1), dtype=np.uint8),
                contour=None,
                morphology=None,
                segmented_full=np.zeros((1, 1, 3), dtype=np.uint8),
                segmented_cropped=np.zeros((1, 1, 3), dtype=np.uint8),
                standardized_grain=np.zeros(self.target_size + (3,), dtype=np.uint8),
                success=False,
                error_message=f"Failed to load image file from {image_path}",
                processing_time_ms=0.0
            )
        return self.process_bgr(
            bgr,
            image_name=image_name,
            label_id=label_id,
            class_name=class_name,
            relative_path=relative_path,
        )

    def save_comparison_plot(
        self,
        res: PreprocessingResult,
        output_path: Path
    ) -> None:
        """
        Generate and save a high-resolution 4-panel visual comparison:
        Original RGB -> Denoised Grayscale -> Binary Mask (with Contour overlay) -> Standardized Segmented Grain.
        """
        fig, axes = plt.subplots(1, 4, figsize=(18, 5.2), facecolor="#0f172a")

        # 1. Original RGB
        axes[0].imshow(res.rgb)
        h, w = res.original_shape[:2]
        axes[0].set_title(f"1. Original RGB\n({w} x {h})", color="#e2e8f0", fontsize=12, fontweight="bold", pad=8)
        axes[0].axis("off")

        # 2. Denoised Grayscale
        axes[1].imshow(res.gray_denoised, cmap="gray")
        axes[1].set_title(f"2. Grayscale (Denoised)\n(Mean: {res.gray_denoised.mean():.1f})", color="#e2e8f0", fontsize=12, fontweight="bold", pad=8)
        axes[1].axis("off")

        # 3. Binary Mask with Contour Overlay
        mask_rgb = cv2.cvtColor(res.binary_mask, cv2.COLOR_GRAY2RGB)
        if res.contour is not None and res.morphology is not None:
            # Draw green contour
            cv2.drawContours(mask_rgb, [res.contour], -1, (0, 255, 100), thickness=2)
            # Draw bounding box in bright cyan
            bx, by, bw, bh = res.morphology.bbox
            cv2.rectangle(mask_rgb, (bx, by), (bx + bw, by + bh), (0, 200, 255), thickness=1)
        axes[2].imshow(mask_rgb)
        area_str = f"Area: {res.morphology.area:,.0f} px" if res.morphology else "N/A"
        axes[2].set_title(f"3. Binary Grain Mask\n({area_str})", color="#e2e8f0", fontsize=12, fontweight="bold", pad=8)
        axes[2].axis("off")

        # 4. Standardized & Segmented Grain (RGB)
        std_rgb = cv2.cvtColor(res.standardized_grain, cv2.COLOR_BGR2RGB)
        axes[3].imshow(std_rgb)
        th, tw = res.processed_shape[:2]
        axes[3].set_title(f"4. Segmented & Standardized\n({tw} x {th} for EfficientNet-B0)", color="#38bdf8", fontsize=12, fontweight="bold", pad=8)
        axes[3].axis("off")

        # Master Title with Class Name and Label
        status_txt = "SUCCESS" if res.success else f"FAILED ({res.error_message})"
        fig.suptitle(
            f"Rice Preprocessing & Grain Segmentation  |  Class: [{res.label_id}] {res.class_name}  |  File: {res.image_name}  [{status_txt}]",
            color="#f8fafc",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.93])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Dataset Loading & Multi-Image Runner
# ---------------------------------------------------------------------------
def load_split_entries(split_filename: str) -> List[Tuple[str, int, str]]:
    """
    Load relative path, label ID, and class folder name from a split text file.
    """
    split_path = DATASETS_DIR / split_filename
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    entries = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                rel_path, label_str = parts
                label_id = int(label_str)
                parts_path = Path(rel_path).parts
                class_folder = parts_path[1] if len(parts_path) >= 2 else "UNKNOWN"
                entries.append((rel_path, label_id, class_folder))
    return entries


def select_balanced_samples(
    entries: List[Tuple[str, int, str]],
    num_samples: int = 20
) -> List[Tuple[str, int, str]]:
    """
    Select representative samples evenly distributed across all classes.
    """
    by_class: Dict[int, List[Tuple[str, int, str]]] = {}
    for entry in entries:
        by_class.setdefault(entry[1], []).append(entry)

    sorted_classes = sorted(by_class.keys())
    num_classes = len(sorted_classes)
    
    # Calculate samples per class
    base_per_class = num_samples // num_classes
    remainder = num_samples % num_classes

    selected = []
    for i, cid in enumerate(sorted_classes):
        take = base_per_class + (1 if i < remainder else 0)
        c_entries = by_class[cid]
        # Pick spaced samples across the class
        step = max(1, len(c_entries) // max(1, take))
        picked = [c_entries[j * step] for j in range(min(take, len(c_entries)))]
        selected.extend(picked)

    return selected[:num_samples]


def run_preprocessing_pipeline(
    split_name: str = "train",
    num_samples: int = 20,
    save_individual_intermediates: bool = True,
    output_dir: Optional[Path] = None,
) -> Tuple[List[PreprocessingResult], Path]:
    """
    Executes preprocessing on representative samples, saves side-by-side comparison images,
    and produces a detailed text report.
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    split_filename = SPLIT_FILES.get(split_name, "rice_train.txt")
    print(f"[*] Loading dataset entries from {split_filename} ...")
    entries = load_split_entries(split_filename)
    print(f"    Loaded {len(entries):,} total entries across splits.")

    selected_entries = select_balanced_samples(entries, num_samples=num_samples)
    print(f"[*] Selected {len(selected_entries)} representative samples across {len(set(e[1] for e in entries))} classes.")

    preprocessor = RicePreprocessor(target_size=DEFAULT_TARGET_SIZE)

    results: List[PreprocessingResult] = []
    failures: List[PreprocessingResult] = []

    visuals_dir = output_dir / "visual_comparisons"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    intermediate_dir = output_dir / "intermediate_samples"
    if save_individual_intermediates:
        intermediate_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Processing images through pipeline ...")
    for idx, (rel_path, label_id, class_name) in enumerate(selected_entries, 1):
        img_path = RICE_IMAGE_ROOT / rel_path
        res = preprocessor.process_image(
            image_path=img_path,
            label_id=label_id,
            class_name=class_name,
            relative_path=rel_path
        )
        results.append(res)

        if not res.success:
            failures.append(res)
            print(f"    [{idx:02d}/{len(selected_entries)}] FAILED: {rel_path} - {res.error_message}")
            continue

        # Save 4-panel visual comparison
        comp_filename = f"sample_{idx:02d}_class_{label_id}_{class_name}_{img_path.stem}.png"
        comp_path = visuals_dir / comp_filename
        preprocessor.save_comparison_plot(res, comp_path)

        # Save individual intermediate images if requested
        if save_individual_intermediates:
            sample_sub = intermediate_dir / f"sample_{idx:02d}_class_{label_id}"
            sample_sub.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(sample_sub / "01_original_bgr.png"), res.original_bgr)
            cv2.imwrite(str(sample_sub / "02_grayscale_denoised.png"), res.gray_denoised)
            cv2.imwrite(str(sample_sub / "03_binary_mask.png"), res.binary_mask)
            cv2.imwrite(str(sample_sub / "04_segmented_full.png"), res.segmented_full)
            cv2.imwrite(str(sample_sub / "05_segmented_cropped.png"), res.segmented_cropped)
            cv2.imwrite(str(sample_sub / "06_standardized_224x224.png"), res.standardized_grain)

        h_orig, w_orig = res.original_shape[:2]
        bw, bh = res.morphology.bbox[2:] if res.morphology else (0, 0)
        area = res.morphology.area if res.morphology else 0
        print(f"    [{idx:02d}/{len(selected_entries)}] OK: Class {label_id} ({class_name:<5s}) | Orig: {w_orig}x{h_orig} -> Grain: {bw}x{bh} (Area: {area:,.0f}px) -> Out: 224x224 in {res.processing_time_ms:.1f}ms")

    # Generate Preprocessing Report
    report_path = output_dir / "preprocessing_report.txt"
    generate_report(results, report_path, preprocessor)
    print(f"[*] Preprocessing report written -> {report_path}")

    # Generate master grid summary image
    grid_path = output_dir / "preprocessing_summary_grid.png"
    create_master_summary_grid(results, grid_path)
    print(f"[*] Preprocessing summary grid saved -> {grid_path}")

    return results, report_path


def generate_report(
    results: List[PreprocessingResult],
    report_path: Path,
    preprocessor: RicePreprocessor
) -> None:
    """
    Write detailed preprocessing and segmentation report.
    """
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    total = len(results)

    # Calculate statistics on input dimensions
    widths = [r.original_shape[1] for r in successful]
    heights = [r.original_shape[0] for r in successful]
    areas = [r.morphology.area for r in successful if r.morphology]
    times = [r.processing_time_ms for r in results]

    lines = [
        "=" * 76,
        "  RICE IMAGE PREPROCESSING & GRAIN SEGMENTATION REPORT",
        f"  Project   : Rice Quality and Defect Assessment using Hybrid ML & Feature Fusion",
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 76,
        "",
        "1. EXECUTIVE SUMMARY",
        f"  - Total Images Processed       : {total}",
        f"  - Successful Segmentations     : {len(successful)} ({len(successful)/total*100:.1f}%)",
        f"  - Failed Segmentations         : {len(failed)} ({len(failed)/total*100:.1f}%)",
        f"  - Average Processing Time      : {np.mean(times):.2f} ms / image",
        "",
        "2. IMAGE DIMENSIONS & RESOLUTION TRANSFORMATION",
        f"  - Original Dimensions (Min)    : {min(widths) if widths else 0} x {min(heights) if heights else 0} px",
        f"  - Original Dimensions (Max)    : {max(widths) if widths else 0} x {max(heights) if heights else 0} px",
        f"  - Original Dimensions (Mean)   : {np.mean(widths) if widths else 0:.1f} x {np.mean(heights) if heights else 0:.1f} px",
        f"  - Standardized Output Size     : {preprocessor.target_size[0]} x {preprocessor.target_size[1]} px (3 Channels, RGB/BGR)",
        f"  - Aspect Ratio Handling        : Preserved with isotropic scaling and centered background padding",
        "",
        "3. SEGMENTATION & PREPROCESSING METHODOLOGY",
        "  - Color Space Conversions      : RGB, Grayscale, HSV, LAB (computed & available for feature extraction)",
        f"  - Noise Reduction              : Gaussian Blur (Kernel: {preprocessor.blur_ksize}, sigma=0) for boundary-preserving smoothing",
        "  - Foreground Separation        : Automatic Otsu Thresholding on denoised grayscale channel",
        f"  - Morphological Operations     : Elliptical Structuring Element (Kernel: {preprocessor.morph_ksize})",
        "                                   * cv2.MORPH_CLOSE to bridge chalky/diseased internal grain voids",
        "                                   * cv2.MORPH_OPEN to eliminate tiny stray dust particles",
        "  - Contour Extraction           : cv2.findContours with cv2.RETR_EXTERNAL",
        "  - Grain Region Isolation       : Largest connected contour filter with filled binary mask generation",
        f"  - Standardized Cropping        : Bounding box crop with {preprocessor.margin_ratio*100:.0f}% margin, padded to {preprocessor.target_size[0]}x{preprocessor.target_size[1]}",
        "",
        "4. GRAIN MORPHOLOGY SUMMARY (PROCESSED SAMPLES)",
        f"  - Grain Area (Min)             : {min(areas) if areas else 0:,.0f} px",
        f"  - Grain Area (Max)             : {max(areas) if areas else 0:,.0f} px",
        f"  - Grain Area (Mean)            : {np.mean(areas) if areas else 0:,.0f} px",
        "",
        "5. PROCESSED SAMPLES BREAKDOWN",
        f"  {'#':<3} {'Class ID':<9} {'Class Name':<12} {'Original (WxH)':<16} {'Grain BBox':<14} {'Area (px)':<12} {'Status':<10}",
        f"  {'-'*3} {'-'*9} {'-'*12} {'-'*16} {'-'*14} {'-'*12} {'-'*10}",
    ]

    for idx, r in enumerate(results, 1):
        if r.success and r.morphology:
            orig_dim = f"{r.original_shape[1]}x{r.original_shape[0]}"
            bbox_dim = f"{r.morphology.bbox[2]}x{r.morphology.bbox[3]}"
            area_str = f"{r.morphology.area:,.0f}"
            status_str = "SUCCESS"
        else:
            orig_dim = f"{r.original_shape[1]}x{r.original_shape[0]}" if r.original_shape[0] > 0 else "N/A"
            bbox_dim = "N/A"
            area_str = "N/A"
            status_str = f"FAIL ({r.error_message})"

        lines.append(f"  {idx:<3} {r.label_id:<9} {r.class_name:<12} {orig_dim:<16} {bbox_dim:<14} {area_str:<12} {status_str:<10}")

    lines.extend([
        "",
        "6. ISSUES & ANOMALIES DETECTED",
    ])

    if failed:
        lines.append(f"  - WARNING: {len(failed)} images failed segmentation.")
        for f in failed:
            lines.append(f"    * {f.relative_path}: {f.error_message}")
    else:
        lines.append("  - None: 100% of the sample rice grains were successfully detected, cleanly segmented from background,")
        lines.append("    and standardized to 224x224 resolution without artifacts.")

    lines.extend([
        "",
        "7. INTEGRATION READINESS",
        "  - Ready for Stage 3 Feature Extraction:",
        "    * Shape Features: Morphological contour, area, perimeter, circularity, aspect ratio, extent, solidity.",
        "    * Texture Features: GLCM (contrast, dissimilarity, homogeneity, energy, correlation, ASM).",
        "    * Colour Features: Mean/Std/Skewness in RGB, HSV, and LAB colour spaces.",
        "    * Deep Features: Standardized 224x224 RGB grain images for EfficientNet-B0 backbone.",
        "",
        "=" * 76,
        "  END OF REPORT",
        "=" * 76,
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def create_master_summary_grid(
    results: List[PreprocessingResult],
    output_path: Path
) -> None:
    """
    Generate a visual grid showing representative segmented standardized grains across all classes.
    """
    successful = [r for r in results if r.success]
    if not successful:
        return

    n = len(successful)
    cols = 5
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.5), facecolor="#090d16")
    axes = np.array(axes).reshape(-1)

    for i in range(len(axes)):
        if i < n:
            r = successful[i]
            img_rgb = cv2.cvtColor(r.standardized_grain, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img_rgb)
            axes[i].set_title(
                f"Class {r.label_id}: {r.class_name}\n{r.original_shape[1]}x{r.original_shape[0]} -> 224x224",
                color="#38bdf8",
                fontsize=9,
                pad=6
            )
            axes[i].axis("off")
        else:
            axes[i].axis("off")

    fig.suptitle(
        f"GrainSet Rice Preprocessing & Segmentation Gallery ({n} Samples)",
        color="#f8fafc",
        fontsize=14,
        fontweight="bold",
        y=0.99
    )

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Rice Image Preprocessing and Grain Segmentation Pipeline"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of representative samples to process (default: 20)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test", "train_bal"],
        help="Dataset split to draw samples from (default: train)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR),
        help="Directory to save visual outputs and report"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  Rice Quality & Defect Assessment - Preprocessing Pipeline")
    print(f"  Samples : {args.samples}")
    print(f"  Split   : {args.split}")
    print(f"  Output  : {args.output_dir}")
    print("=" * 70)

    results, report_path = run_preprocessing_pipeline(
        split_name=args.split,
        num_samples=args.samples,
        output_dir=Path(args.output_dir)
    )

    print("\n" + "=" * 70)
    print("  PREPROCESSING COMPLETED SUCCESSFULLY")
    print(f"  Processed : {len(results)} images")
    print(f"  Report    : {report_path}")
    print(f"  Outputs   : {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
