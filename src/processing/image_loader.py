"""
Image Loader and Validator for PCB Fault Detection.

Provides robust validation, decoding from multiple input sources (file path,
bytes, PIL Image, numpy array), and safe writing to disk.
"""

import io
import logging
from pathlib import Path
from typing import Any, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def validate_and_load_image(
    image_source: Union[str, Path, bytes, np.ndarray, Image.Image]
) -> Tuple[bool, Optional[np.ndarray], Optional[str]]:
    """
    Validate and decode an image from various input types into a BGR numpy ndarray.

    Args:
        image_source: File path, Path object, raw bytes, numpy ndarray, or PIL Image.

    Returns:
        Tuple of (success: bool, image_bgr: Optional[np.ndarray], error_message: Optional[str]).
    """
    if image_source is None:
        return False, None, "Image source is None."

    # 1. Input is already a numpy ndarray
    if isinstance(image_source, np.ndarray):
        if image_source.size == 0:
            return False, None, "Empty numpy array provided."
        if image_source.ndim == 2:
            # Grayscale to BGR
            bgr = cv2.cvtColor(image_source, cv2.COLOR_GRAY2BGR)
            return True, bgr, None
        elif image_source.ndim == 3:
            if image_source.shape[2] == 4:
                # BGRA to BGR
                bgr = cv2.cvtColor(image_source, cv2.COLOR_BGRA2BGR)
                return True, bgr, None
            elif image_source.shape[2] == 3:
                return True, image_source.copy(), None
            else:
                return False, None, f"Unsupported channel count: {image_source.shape[2]}"
        return False, None, f"Unsupported numpy dimensions: {image_source.ndim}"

    # 2. Input is a PIL Image
    if isinstance(image_source, Image.Image):
        try:
            rgb = np.array(image_source.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return True, bgr, None
        except Exception as e:
            return False, None, f"Failed to convert PIL Image to ndarray: {e}"

    # 3. Input is raw bytes
    if isinstance(image_source, bytes):
        if len(image_source) == 0:
            return False, None, "Empty byte buffer provided."
        try:
            nparr = np.frombuffer(image_source, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return False, None, "Failed to decode image bytes with OpenCV."
            return True, img, None
        except Exception as e:
            return False, None, f"Exception decoding image bytes: {e}"

    # 4. Input is a file path (str or Path)
    if isinstance(image_source, (str, Path)):
        path = Path(image_source)
        if not path.is_file():
            return False, None, f"Image file not found: {path}"
        try:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                # Attempt decoding via PIL as fallback (e.g. unicode path on Windows)
                try:
                    pil_img = Image.open(path)
                    rgb = np.array(pil_img.convert("RGB"))
                    return True, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), None
                except Exception:
                    return False, None, f"Failed to decode image file: {path}"
            return True, img, None
        except Exception as e:
            return False, None, f"Error reading image file {path}: {e}"

    return False, None, f"Unsupported image source type: {type(image_source)}"


def save_image(image_bgr: np.ndarray, output_path: Union[str, Path]) -> bool:
    """
    Save a BGR numpy image to disk, creating parent directories if needed.

    Args:
        image_bgr: Input BGR numpy image.
        output_path: Destination path.

    Returns:
        True if written successfully, False otherwise.
    """
    if image_bgr is None or image_bgr.size == 0:
        return False
    try:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(dest), image_bgr))
    except Exception as e:
        logger.error(f"Failed to save image to {output_path}: {e}")
        return False
