from __future__ import annotations

from typing import Union

import numpy as np
from PIL import Image

from src.inference.rice_gate import RiceGateResult, SETUP_MESSAGE, validate_rice_or_reject


def validate_rice_image(image_input: Union[Image.Image, np.ndarray]) -> RiceGateResult:
    """Validate webcam/upload input using the trained rice-vs-non-rice gate when available."""
    return validate_rice_or_reject(image_input)


__all__ = ["validate_rice_image", "RiceGateResult", "SETUP_MESSAGE"]
