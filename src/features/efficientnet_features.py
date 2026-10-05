"""EfficientNet-B0 deep feature extraction utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from src.preprocessing.preprocess import RicePreprocessor


@dataclass(frozen=True)
class ImageEntry:
    """One manifest entry with its original split metadata."""

    image_path: str
    split: str
    class_id: int
    class_name: str


@dataclass
class ExtractionResult:
    """Features and extraction diagnostics for one split."""

    entries: List[ImageEntry]
    features: np.ndarray
    failed_images: List[Tuple[ImageEntry, str]]
    elapsed_seconds: float


def load_efficientnet_b0(device: torch.device) -> Tuple[torch.nn.Module, int, str]:
    """Load ImageNet-pretrained EfficientNet-B0 as an inference-only extractor."""
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    model.classifier = torch.nn.Identity()
    model.eval().to(device)
    with torch.inference_mode():
        feature_dimension = int(model(torch.zeros(1, 3, 224, 224, device=device)).shape[1])
    return model, feature_dimension, str(weights)


def image_to_tensor(
    image_bgr: np.ndarray,
    image_size: int = 224,
) -> torch.Tensor:
    """Convert a standardized BGR image to an ImageNet-normalized RGB tensor."""
    if image_bgr.shape[:2] != (image_size, image_size):
        raise ValueError(f"Expected {image_size}x{image_size} image, got {image_bgr.shape[:2]}")
    image_rgb = image_bgr[:, :, ::-1]
    image = Image.fromarray(image_rgb)
    weights = EfficientNet_B0_Weights.DEFAULT
    return weights.transforms()(image)


def extract_split_features(
    entries: Sequence[ImageEntry],
    image_root: Path,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 32,
    preprocessor: RicePreprocessor | None = None,
) -> ExtractionResult:
    """Segment images, extract features in batches, and retain manifest order."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    preprocessor = preprocessor or RicePreprocessor(target_size=(224, 224))
    tensors: List[torch.Tensor] = []
    valid_entries: List[ImageEntry] = []
    failures: List[Tuple[ImageEntry, str]] = []
    feature_batches: List[np.ndarray] = []
    start = time.perf_counter()

    def flush_batch() -> None:
        if not tensors:
            return
        batch = torch.stack(tensors).to(device)
        with torch.inference_mode():
            output = model(batch).detach().cpu().numpy()
        feature_batches.append(output.astype(np.float32, copy=False))
        tensors.clear()

    for index, entry in enumerate(entries, start=1):
        result = preprocessor.process_image(
            image_root / entry.image_path,
            label_id=entry.class_id,
            class_name=entry.class_name,
            relative_path=entry.image_path,
        )
        if not result.success:
            failures.append((entry, result.error_message or "Unknown preprocessing failure"))
            continue
        try:
            tensors.append(image_to_tensor(result.standardized_grain))
            valid_entries.append(entry)
        except Exception as exc:
            failures.append((entry, f"Tensor conversion failed: {exc}"))
        if len(tensors) >= batch_size:
            flush_batch()
        if index % 500 == 0 or index == len(entries):
            print(f"    processed {index:,}/{len(entries):,} images")
    flush_batch()

    features = np.vstack(feature_batches) if feature_batches else np.empty((0, 0), dtype=np.float32)
    if len(valid_entries) != len(features):
        raise RuntimeError("Feature rows are not aligned with valid manifest entries")
    return ExtractionResult(valid_entries, features, failures, time.perf_counter() - start)
