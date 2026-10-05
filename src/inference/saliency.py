"""Visual saliency and activation mapping for EfficientNet-B0 backbone.

Computes spatial activation heatmaps on standardized rice grain images to visually
highlight the regions (e.g., fungal spots, embryo damage, fissures, chalky areas)
that influenced the deep feature representation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

from src.features.efficientnet_features import image_to_tensor
from src.inference.predict import load_efficientnet, preprocess_for_inference


def generate_activation_heatmap(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.5,
) -> Dict[str, Any]:
    """Generate a spatial activation heatmap overlaid on the standardized rice grain.

    Args:
        image_input: Path, PIL Image, or BGR/RGB numpy array.
        colormap: OpenCV colormap constant (default: COLORMAP_JET).
        alpha: Heatmap blend weight between 0.0 and 1.0 (default: 0.5).

    Returns:
        Dict containing:
            - 'original_rgb': Standardized grain RGB array (224x224).
            - 'heatmap_rgb': Colorized activation heatmap (224x224).
            - 'overlay_rgb': Superimposed heatmap over the grain (224x224).
            - 'peak_intensity': Maximum spatial activation value.
            - 'focus_area_ratio': Fraction of grain surface with above-average activation.
    """
    processed = preprocess_for_inference(image_input)
    standardized_bgr = processed["standardized_grain"]
    standardized_rgb = processed["preprocessed_rgb"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_efficientnet().to(device)

    # Convert to normalized tensor
    tensor = image_to_tensor(standardized_bgr, image_size=224).unsqueeze(0).to(device)

    # Extract spatial feature map from the convolutional backbone
    # In torchvision efficientnet_b0, model.features produces (1, 1280, 7, 7)
    with torch.inference_mode():
        if hasattr(model, "features"):
            features = model.features(tensor)
        else:
            # Fallback if backbone wrapped differently
            features = model(tensor).reshape(1, -1, 1, 1)

    # Compute spatial activation: mean across all 1280 feature channels
    spatial_map = torch.mean(features, dim=1).squeeze(0).cpu().numpy()

    # ReLu on activations to keep only positive contributions
    spatial_map = np.maximum(spatial_map, 0)

    # Normalize to 0.0 - 1.0
    map_min = float(spatial_map.min())
    map_max = float(spatial_map.max())
    if map_max > map_min:
        norm_map = (spatial_map - map_min) / (map_max - map_min)
    else:
        norm_map = np.zeros_like(spatial_map, dtype=np.float32)

    # Upsample the 7x7 spatial activation map to 224x224
    upsampled = cv2.resize(norm_map, (224, 224), interpolation=cv2.INTER_CUBIC)
    upsampled = np.clip(upsampled, 0.0, 1.0)

    # Apply grain binary mask so heatmap only illuminates the grain, not the black background
    grain_mask = (cv2.cvtColor(standardized_bgr, cv2.COLOR_BGR2GRAY) > 10).astype(np.float32)
    upsampled_masked = upsampled * grain_mask

    # Colorize heatmap
    heatmap_uint8 = np.uint8(255 * upsampled_masked)
    heatmap_color_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)

    # Blend with original standardized grain
    overlay_rgb = np.uint8(
        standardized_rgb * (1.0 - alpha * grain_mask[:, :, None])
        + heatmap_color_rgb * (alpha * grain_mask[:, :, None])
    )

    # Mask out background completely for clean presentation
    overlay_rgb[grain_mask == 0] = 0

    peak_intensity = float(upsampled.max())
    focus_area_ratio = float(np.mean(upsampled_masked > 0.5))

    return {
        "original_rgb": standardized_rgb,
        "heatmap_rgb": heatmap_color_rgb,
        "overlay_rgb": overlay_rgb,
        "peak_intensity": round(peak_intensity, 4),
        "focus_area_ratio": round(focus_area_ratio, 4),
    }


__all__ = ["generate_activation_heatmap"]
