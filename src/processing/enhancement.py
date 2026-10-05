"""
PCB Image Enhancement Module.

Provides aspect-ratio-preserving resizing, LAB-CLAHE contrast/illumination
equalization, and edge-preserving bilateral filtering for PCB defect inspection.
"""

from typing import Any, Dict, Tuple
import cv2
import numpy as np


def resize_preserve_aspect(
    image: np.ndarray, max_dimension: int = 2048
) -> Tuple[np.ndarray, float]:
    """
    Resize an image if its longest side exceeds max_dimension, preserving aspect ratio.

    Args:
        image: Input image as numpy ndarray.
        max_dimension: Maximum allowed width or height in pixels.

    Returns:
        Tuple of (resized_image, scale_factor).
    """
    height, width = image.shape[:2]
    max_side = max(height, width)

    if max_side <= max_dimension:
        return image.copy(), 1.0

    scale = max_dimension / float(max_side)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def enhance_pcb_illumination(image_bgr: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Apply illumination and contrast enhancement for PCB imagery.
    Converts to LAB color space and applies CLAHE on the L-channel to balance
    uneven lighting and reflections on solder joints without blowing out details.
    """
    if image_bgr.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        return clahe.apply(image_bgr)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)

    merged_lab = cv2.merge([enhanced_l, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


def reduce_pcb_noise(
    image_bgr: np.ndarray, d: int = 5, sigma_color: float = 35, sigma_space: float = 35
) -> np.ndarray:
    """
    Apply controlled edge-preserving bilateral filtering to reduce noise without
    blurring fine electronic components or copper tracks.
    """
    return cv2.bilateralFilter(image_bgr, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR or single channel image to uint8 grayscale."""
    if image_bgr.ndim == 2:
        return image_bgr.copy()
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def enhance_pcb_pipeline(
    image_bgr: np.ndarray, max_dimension: int = 2048
) -> Dict[str, Any]:
    """
    Execute full enhancement workflow: resize -> contrast equalize -> denoise -> grayscale.

    Args:
        image_bgr: Input BGR image ndarray.
        max_dimension: Max allowable dimension.

    Returns:
        Dict containing processed ndarrays and metadata.
    """
    resized_bgr, scale = resize_preserve_aspect(image_bgr, max_dimension=max_dimension)
    enhanced_bgr = enhance_pcb_illumination(resized_bgr)
    denoised_bgr = reduce_pcb_noise(enhanced_bgr)
    grayscale = to_grayscale(denoised_bgr)

    return {
        "original_image": image_bgr,
        "resized_image": resized_bgr,
        "enhanced_image": enhanced_bgr,
        "processed_image": denoised_bgr,
        "grayscale_image": grayscale,
        "scale_factor": scale,
        "dimensions": denoised_bgr.shape,
    }
