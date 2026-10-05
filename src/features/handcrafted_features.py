"""
handcrafted_features.py
=======================
Handcrafted Feature Extraction Pipeline for Rice Grain Images

Project:
    Rice Quality and Defect Assessment using Hybrid Machine Learning and Deep Feature Fusion

Description:
    Extracts 62 domain-specific handcrafted features from segmented rice grains across 3 distinct groups:
    1. Shape / Morphological Features (14 features)
    2. GLCM Texture Features on grain region (12 features)
    3. Colour Statistics across RGB, HSV, and LAB spaces on grain region (36 features)

Features are extracted strictly from the grain region (excluding background pixels).
Numerically stable calculations protect against division by zero, empty contours, and edge cases.
"""

import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import cv2
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. SHAPE FEATURES
# ---------------------------------------------------------------------------
def extract_shape_features(contour: Optional[np.ndarray], mask_shape: Tuple[int, int]) -> Dict[str, float]:
    """
    Extract geometric and morphological features from the binary grain contour.

    Args:
        contour: Largest OpenCV contour representing the primary rice grain.
        mask_shape: (height, width) of the image/mask.

    Returns:
        Dictionary of 14 shape features.
    """
    default_features = {
        "shape_area": 0.0,
        "shape_perimeter": 0.0,
        "shape_width": 0.0,
        "shape_height": 0.0,
        "shape_aspect_ratio": 0.0,
        "shape_extent": 0.0,
        "shape_solidity": 0.0,
        "shape_circularity": 0.0,
        "shape_eccentricity": 0.0,
        "shape_major_axis_length": 0.0,
        "shape_minor_axis_length": 0.0,
        "shape_equivalent_diameter": 0.0,
        "shape_convex_hull_area": 0.0,
        "shape_bbox_area": 0.0,
    }

    if contour is None or len(contour) < 3:
        return default_features

    area = float(cv2.contourArea(contour))
    if area <= 0.0:
        return default_features

    perimeter = float(cv2.arcLength(contour, closed=True))
    x, y, w, h = cv2.boundingRect(contour)
    w_f, h_f = float(w), float(h)
    bbox_area = w_f * h_f
    aspect_ratio = w_f / h_f if h_f > 0.0 else 0.0
    extent = area / bbox_area if bbox_area > 0.0 else 0.0

    # Convex Hull & Solidity
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0.0 else 0.0

    # Circularity = 4 * pi * Area / (Perimeter^2)
    circularity = (4.0 * math.pi * area) / (perimeter ** 2) if perimeter > 0.0 else 0.0
    circularity = min(1.0, max(0.0, circularity))  # Clamp to [0, 1]

    # Equivalent Diameter = sqrt(4 * Area / pi)
    eq_diameter = math.sqrt(4.0 * area / math.pi)

    # Major and Minor Axis Lengths & Eccentricity
    # Use cv2.fitEllipse when contour has >= 5 points, else fallback to minAreaRect
    if len(contour) >= 5:
        try:
            (center, (axis_1, axis_2), angle) = cv2.fitEllipse(contour)
            minor_axis = min(axis_1, axis_2)
            major_axis = max(axis_1, axis_2)
        except Exception:
            rect = cv2.minAreaRect(contour)
            dim1, dim2 = rect[1]
            minor_axis, major_axis = min(dim1, dim2), max(dim1, dim2)
    else:
        rect = cv2.minAreaRect(contour)
        dim1, dim2 = rect[1]
        minor_axis, major_axis = min(dim1, dim2), max(dim1, dim2)

    minor_axis = float(minor_axis)
    major_axis = float(major_axis)

    # Eccentricity = sqrt(1 - (b/a)^2)
    if major_axis > 0.0 and minor_axis <= major_axis:
        eccentricity = math.sqrt(max(0.0, 1.0 - (minor_axis / major_axis) ** 2))
    else:
        eccentricity = 0.0

    return {
        "shape_area": area,
        "shape_perimeter": perimeter,
        "shape_width": w_f,
        "shape_height": h_f,
        "shape_aspect_ratio": aspect_ratio,
        "shape_extent": extent,
        "shape_solidity": solidity,
        "shape_circularity": circularity,
        "shape_eccentricity": eccentricity,
        "shape_major_axis_length": major_axis,
        "shape_minor_axis_length": minor_axis,
        "shape_equivalent_diameter": eq_diameter,
        "shape_convex_hull_area": hull_area,
        "shape_bbox_area": bbox_area,
    }


# ---------------------------------------------------------------------------
# 2. GLCM TEXTURE FEATURES (Grain Region Only)
# ---------------------------------------------------------------------------
def extract_glcm_features(
    gray_image: np.ndarray,
    mask: np.ndarray,
    distances: Tuple[int, ...] = (1, 2),
    angles: Tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    levels: int = 32
) -> Dict[str, float]:
    """
    Extract Gray-Level Co-occurrence Matrix (GLCM) texture statistics.
    Crucially computes co-occurrences exclusively between pixel pairs belonging to the grain.

    Args:
        gray_image: 2D grayscale image (uint8, 0-255).
        mask: 2D binary mask (uint8, >0 for grain, 0 for background).
        distances: Pixel displacement distances (e.g., 1 and 2 pixels).
        angles: Orientation angles in radians (0, 45, 90, 135 degrees).
        levels: Number of quantized intensity bins (32 levels for robust density).

    Returns:
        Dictionary of 12 GLCM texture features (mean and std for 6 properties).
    """
    default_features = {
        "glcm_contrast_mean": 0.0,
        "glcm_contrast_std": 0.0,
        "glcm_dissimilarity_mean": 0.0,
        "glcm_dissimilarity_std": 0.0,
        "glcm_homogeneity_mean": 0.0,
        "glcm_homogeneity_std": 0.0,
        "glcm_energy_mean": 0.0,
        "glcm_energy_std": 0.0,
        "glcm_correlation_mean": 0.0,
        "glcm_correlation_std": 0.0,
        "glcm_asm_mean": 0.0,
        "glcm_asm_std": 0.0,
    }

    if gray_image is None or mask is None or np.sum(mask > 0) == 0:
        return default_features

    # Quantize gray intensities into specified levels
    step = 256 // levels
    quantized = np.clip(gray_image // step, 0, levels - 1).astype(np.int32)
    valid_mask = (mask > 0)

    # Generate displacement (dr, dc) offsets for all distance-angle pairs
    # Angle 0 deg (0 rad):       (0, d)
    # Angle 45 deg (pi/4 rad):   (-d, d)
    # Angle 90 deg (pi/2 rad):   (-d, 0)
    # Angle 135 deg (3*pi/4 rad):(-d, -d)
    offsets: List[Tuple[int, int]] = []
    for d in distances:
        offsets.append((0, d))
        offsets.append((-d, d))
        offsets.append((-d, 0))
        offsets.append((-d, -d))

    H, W = quantized.shape
    i_ind, j_ind = np.indices((levels, levels))
    diff_sq = (i_ind - j_ind) ** 2
    diff_abs = np.abs(i_ind - j_ind)

    contrast_list: List[float] = []
    dissimilarity_list: List[float] = []
    homogeneity_list: List[float] = []
    energy_list: List[float] = []
    correlation_list: List[float] = []
    asm_list: List[float] = []

    for dr, dc in offsets:
        r_start, r_end = max(0, -dr), min(H, H - dr)
        c_start, c_end = max(0, -dc), min(W, W - dc)

        src_vals = quantized[r_start:r_end, c_start:c_end]
        src_mask = valid_mask[r_start:r_end, c_start:c_end]

        tgt_vals = quantized[r_start + dr:r_end + dr, c_start + dc:c_end + dc]
        tgt_mask = valid_mask[r_start + dr:r_end + dr, c_start + dc:c_end + dc]

        # Valid pairs must both reside inside the grain mask
        pair_mask = src_mask & tgt_mask
        valid_src = src_vals[pair_mask]
        valid_tgt = tgt_vals[pair_mask]

        if len(valid_src) == 0:
            continue

        # Compute symmetric 2D co-occurrence matrix
        P_mat = np.zeros((levels, levels), dtype=np.float64)
        np.add.at(P_mat, (valid_src, valid_tgt), 1.0)
        np.add.at(P_mat, (valid_tgt, valid_src), 1.0)

        total_pairs = np.sum(P_mat)
        if total_pairs <= 0.0:
            continue

        P = P_mat / total_pairs

        # 1. Contrast: sum((i - j)^2 * P(i, j))
        contrast = float(np.sum(diff_sq * P))
        contrast_list.append(contrast)

        # 2. Dissimilarity: sum(|i - j| * P(i, j))
        dissimilarity = float(np.sum(diff_abs * P))
        dissimilarity_list.append(dissimilarity)

        # 3. Homogeneity: sum(P(i, j) / (1 + (i - j)^2))
        homogeneity = float(np.sum(P / (1.0 + diff_sq)))
        homogeneity_list.append(homogeneity)

        # 4. ASM (Angular Second Moment): sum(P(i, j)^2) & Energy: sqrt(ASM)
        asm = float(np.sum(P ** 2))
        energy = float(math.sqrt(asm))
        asm_list.append(asm)
        energy_list.append(energy)

        # 5. Correlation: sum((i - mu_i) * (j - mu_j) * P(i, j)) / (std_i * std_j)
        p_x = np.sum(P, axis=1)
        p_y = np.sum(P, axis=0)
        idx_arr = np.arange(levels, dtype=np.float64)

        mu_x = np.sum(idx_arr * p_x)
        mu_y = np.sum(idx_arr * p_y)

        var_x = np.sum(((idx_arr - mu_x) ** 2) * p_x)
        var_y = np.sum(((idx_arr - mu_y) ** 2) * p_y)

        std_x = math.sqrt(max(0.0, var_x))
        std_y = math.sqrt(max(0.0, var_y))

        if std_x * std_y > 1e-10:
            corr = float(np.sum((i_ind - mu_x) * (j_ind - mu_y) * P) / (std_x * std_y))
            # Clip numerical noise
            corr = max(-1.0, min(1.0, corr))
        else:
            corr = 1.0
        correlation_list.append(corr)

    if not contrast_list:
        return default_features

    return {
        "glcm_contrast_mean": float(np.mean(contrast_list)),
        "glcm_contrast_std": float(np.std(contrast_list)),
        "glcm_dissimilarity_mean": float(np.mean(dissimilarity_list)),
        "glcm_dissimilarity_std": float(np.std(dissimilarity_list)),
        "glcm_homogeneity_mean": float(np.mean(homogeneity_list)),
        "glcm_homogeneity_std": float(np.std(homogeneity_list)),
        "glcm_energy_mean": float(np.mean(energy_list)),
        "glcm_energy_std": float(np.std(energy_list)),
        "glcm_correlation_mean": float(np.mean(correlation_list)),
        "glcm_correlation_std": float(np.std(correlation_list)),
        "glcm_asm_mean": float(np.mean(asm_list)),
        "glcm_asm_std": float(np.std(asm_list)),
    }


# ---------------------------------------------------------------------------
# 3. COLOUR FEATURES (Grain Region Only)
# ---------------------------------------------------------------------------
def extract_color_features(bgr_image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Extract color statistical descriptors (Mean, Std, Min, Max) across RGB, HSV, and LAB
    color spaces strictly for the rice-grain pixels (mask > 0).

    Args:
        bgr_image: 3-channel BGR image.
        mask: 2D binary mask.

    Returns:
        Dictionary of 36 color features (9 channels * 4 statistics).
    """
    spaces_config = [
        ("rgb", cv2.COLOR_BGR2RGB, ["r", "g", "b"]),
        ("hsv", cv2.COLOR_BGR2HSV, ["h", "s", "v"]),
        ("lab", cv2.COLOR_BGR2LAB, ["l", "a", "b"]),
    ]

    features: Dict[str, float] = {}

    fg_indices = np.where(mask > 0)
    has_foreground = len(fg_indices[0]) > 0

    for space_name, cvt_code, channel_names in spaces_config:
        converted = cv2.cvtColor(bgr_image, cvt_code)
        for ch_idx, ch_name in enumerate(channel_names):
            prefix = f"color_{space_name}_{ch_name}"
            if has_foreground:
                pixel_vals = converted[fg_indices[0], fg_indices[1], ch_idx].astype(np.float64)
                features[f"{prefix}_mean"] = float(np.mean(pixel_vals))
                features[f"{prefix}_std"] = float(np.std(pixel_vals))
                features[f"{prefix}_min"] = float(np.min(pixel_vals))
                features[f"{prefix}_max"] = float(np.max(pixel_vals))
            else:
                features[f"{prefix}_mean"] = 0.0
                features[f"{prefix}_std"] = 0.0
                features[f"{prefix}_min"] = 0.0
                features[f"{prefix}_max"] = 0.0

    return features


# ---------------------------------------------------------------------------
# 4. COMPLETE SINGLE-IMAGE FEATURE EXTRACTION
# ---------------------------------------------------------------------------
def extract_handcrafted_features(
    bgr_image: np.ndarray,
    mask: np.ndarray,
    contour: Optional[np.ndarray]
) -> Dict[str, float]:
    """
    Extract all 62 handcrafted features for an image given its segmented grain mask and contour.

    Returns:
        Dictionary containing:
        - 14 Shape features
        - 12 GLCM Texture features
        - 36 Colour features
        (Total: 62 features)
    """
    # 1. Shape features
    shape_feats = extract_shape_features(contour, bgr_image.shape[:2])

    # 2. GLCM texture features
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    texture_feats = extract_glcm_features(gray, mask)

    # 3. Colour features
    color_feats = extract_color_features(bgr_image, mask)

    return {**shape_feats, **texture_feats, **color_feats}


def process_image_file_worker(
    item: Tuple[str, int, str, str, str]
) -> Dict[str, Any]:
    """
    Top-level worker function for parallel multiprocessing execution.

    Args:
        item: Tuple of (rel_path, label_id, class_name, split, image_root_str)

    Returns:
        Dictionary with metadata and all extracted features, or error flag.
    """
    rel_path, label_id, class_name, split, image_root_str = item
    img_path = Path(image_root_str) / rel_path

    row_data: Dict[str, Any] = {
        "image_path": rel_path.replace("\\", "/"),
        "split": split,
        "class_id": label_id,
        "class_name": class_name,
        "status": "SUCCESS",
        "error_msg": "",
    }

    try:
        # Load image safely
        img_bytes = np.fromfile(str(img_path), dtype=np.uint8)
        bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if bgr is None:
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

        if bgr is None:
            row_data["status"] = "FAILED"
            row_data["error_msg"] = f"Failed to load image: {rel_path}"
            return row_data

        # Preprocessing & Grain Segmentation
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            row_data["status"] = "FAILED"
            row_data["error_msg"] = "No contours found"
            return row_data

        largest_c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_c)
        if area < 100.0:
            row_data["status"] = "FAILED"
            row_data["error_msg"] = f"Contour area too small ({area:.1f}px)"
            return row_data

        clean_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(clean_mask, [largest_c], -1, 255, thickness=cv2.FILLED)

        # Feature Extraction
        features = extract_handcrafted_features(bgr, clean_mask, largest_c)
        row_data.update(features)

    except Exception as exc:
        row_data["status"] = "FAILED"
        row_data["error_msg"] = str(exc)

    return row_data


# ---------------------------------------------------------------------------
# 5. FEATURE VALIDATION & AUDITING
# ---------------------------------------------------------------------------
def validate_feature_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform rigorous feature validation and quality auditing on extracted DataFrame:
    - Checks for NaN / Null values
    - Checks for Infinite values
    - Identifies constant / zero-variance features
    - Computes feature summary ranges (min, max, mean, std)
    - Detects highly correlated feature pairs (|r| > 0.95)
    """
    meta_cols = ["image_path", "split", "class_id", "class_name", "status", "error_msg"]
    feat_cols = [c for c in df.columns if c not in meta_cols]

    shape_cols = [c for c in feat_cols if c.startswith("shape_")]
    texture_cols = [c for c in feat_cols if c.startswith("glcm_")]
    color_cols = [c for c in feat_cols if c.startswith("color_")]

    feat_matrix = df[feat_cols]

    nan_counts = feat_matrix.isna().sum().to_dict()
    total_nans = sum(nan_counts.values())

    inf_counts = np.isinf(feat_matrix.select_dtypes(include=np.number)).sum().to_dict()
    total_infs = sum(inf_counts.values())

    # Check constant features (std == 0)
    std_series = feat_matrix.std(numeric_only=True)
    constant_features = std_series[std_series == 0].index.tolist()

    # Feature ranges
    feature_ranges = {}
    for col in feat_cols:
        col_data = feat_matrix[col].dropna()
        if len(col_data) > 0:
            feature_ranges[col] = {
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "mean": float(col_data.mean()),
                "std": float(col_data.std()),
            }

    # Highly correlated feature pairs (|r| > 0.95)
    corr_matrix = feat_matrix.corr(numeric_only=True).abs()
    high_corr_pairs = []
    columns = corr_matrix.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            r_val = corr_matrix.iloc[i, j]
            if r_val >= 0.95:
                high_corr_pairs.append((columns[i], columns[j], float(r_val)))

    return {
        "total_images": len(df),
        "total_features": len(feat_cols),
        "shape_feature_count": len(shape_cols),
        "texture_feature_count": len(texture_cols),
        "color_feature_count": len(color_cols),
        "total_nans": total_nans,
        "total_infs": total_infs,
        "constant_features": constant_features,
        "high_corr_pairs": high_corr_pairs,
        "feature_ranges": feature_ranges,
        "feat_cols": feat_cols,
        "shape_cols": shape_cols,
        "texture_cols": texture_cols,
        "color_cols": color_cols,
    }
