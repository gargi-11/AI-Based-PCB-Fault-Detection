"""
PCB Image Preprocessing Module.

Provides robust image ingestion, dimension normalization, mild contrast adjustment,
noise attenuation, and grayscale conversion for downstream PCB fault detection.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
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


def enhance_pcb_illumination(image_bgr: np.ndarray) -> np.ndarray:
    """
    Apply mild illumination and contrast enhancement suitable for PCB imagery.
    Converts to LAB color space and applies gentle CLAHE on the L-channel
    to balance uneven lighting and highlights on solder joints without blowing out details.

    Args:
        image_bgr: Input BGR color image.

    Returns:
        Enhanced BGR color image.
    """
    # Convert BGR to LAB color space
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Conservative CLAHE: clipLimit=2.0, tileGridSize=(8, 8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)

    # Merge channels and convert back to BGR
    merged_lab = cv2.merge([enhanced_l, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


def reduce_pcb_noise(image_bgr: np.ndarray) -> np.ndarray:
    """
    Apply controlled edge-preserving noise reduction without aggressively
    blurring small electronic components or fine copper tracks.
    Uses bilateral filtering with gentle parameters.

    Args:
        image_bgr: Input BGR color image.

    Returns:
        Filtered BGR image.
    """
    # d=5: diameter of pixel neighborhood
    # sigmaColor=35: mild filter sigma in color space to preserve copper track edges
    # sigmaSpace=35: mild filter sigma in coordinate space
    denoised = cv2.bilateralFilter(image_bgr, d=5, sigmaColor=35, sigmaSpace=35)
    return denoised


def preprocess_pcb_image(
    image_path: str,
    output_path: Optional[str] = None,
    max_dimension: int = 2048,
) -> Dict[str, Any]:
    """
    Preprocess a PCB photograph for downstream computer vision and defect analysis.

    Pipeline:
    1. Validates file path existence and readability.
    2. Loads image with OpenCV (BGR).
    3. Preserves a pristine copy of original image.
    4. Resizes if image exceeds max_dimension while preserving aspect ratio.
    5. Performs mild illumination/contrast enhancement via LAB-CLAHE.
    6. Applies edge-preserving bilateral filtering to reduce noise without blurring tracks.
    7. Generates single-channel grayscale representation.
    8. Optionally saves processed color image to disk if output_path is specified.

    Args:
        image_path: Path to input image file.
        output_path: Optional file path to save the processed image.
        max_dimension: Maximum allowed width or height in pixels (default: 2048).

    Returns:
        Dictionary containing:
        - "success": bool indicating whether processing completed successfully
        - "original_image": np.ndarray (or None on failure)
        - "processed_image": np.ndarray (or None on failure)
        - "grayscale_image": np.ndarray (or None on failure)
        - "original_dimensions": Tuple[int, int, int] (height, width, channels)
        - "processed_dimensions": Tuple[int, int, int] (height, width, channels)
        - "metadata": Dict containing scaling factor, operations applied, and file paths
        - "error": Optional error message if processing failed
    """
    path = Path(image_path)

    # 1. Validate file exists
    if not path.is_file():
        return {
            "success": False,
            "original_image": None,
            "processed_image": None,
            "grayscale_image": None,
            "original_dimensions": None,
            "processed_dimensions": None,
            "metadata": {},
            "error": f"Image file not found: {image_path}",
        }

    # 2. Load image using OpenCV
    try:
        raw_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if raw_bgr is None:
            return {
                "success": False,
                "original_image": None,
                "processed_image": None,
                "grayscale_image": None,
                "original_dimensions": None,
                "processed_dimensions": None,
                "metadata": {},
                "error": f"Failed to decode or parse image file: {image_path}",
            }
    except Exception as e:
        return {
            "success": False,
            "original_image": None,
            "processed_image": None,
            "grayscale_image": None,
            "original_dimensions": None,
            "processed_dimensions": None,
            "metadata": {},
            "error": f"Exception while reading image: {str(e)}",
        }

    # 3. Preserve original copy and dimensions
    original_image = raw_bgr.copy()
    orig_h, orig_w = raw_bgr.shape[:2]
    orig_c = raw_bgr.shape[2] if raw_bgr.ndim == 3 else 1
    orig_dims = (orig_h, orig_w, orig_c)

    # 4. Aspect-ratio preserving resize if needed
    resized_bgr, scale_factor = resize_preserve_aspect(raw_bgr, max_dimension=max_dimension)

    # 5. Mild illumination & contrast enhancement
    enhanced_bgr = enhance_pcb_illumination(resized_bgr)

    # 6. Controlled edge-preserving noise reduction
    denoised_bgr = reduce_pcb_noise(enhanced_bgr)

    processed_image = denoised_bgr

    # 7. Grayscale representation
    grayscale_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

    proc_h, proc_w = processed_image.shape[:2]
    proc_c = processed_image.shape[2] if processed_image.ndim == 3 else 1
    proc_dims = (proc_h, proc_w, proc_c)

    metadata: Dict[str, Any] = {
        "input_path": str(path.resolve()),
        "output_path": str(Path(output_path).resolve()) if output_path else None,
        "is_resized": bool(scale_factor < 1.0),
        "scale_factor": float(scale_factor),
        "operations_applied": [
            "aspect_preserving_resize" if scale_factor < 1.0 else "no_resize_needed",
            "lab_clahe_contrast_enhancement",
            "bilateral_edge_preserving_denoise",
            "bgr_to_grayscale_conversion",
        ],
    }

    # 8. Save output if requested
    if output_path:
        try:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_file), processed_image)
            metadata["saved_successfully"] = True
        except Exception as e:
            metadata["saved_successfully"] = False
            metadata["save_error"] = str(e)

    return {
        "success": True,
        "original_image": original_image,
        "processed_image": processed_image,
        "grayscale_image": grayscale_image,
        "original_dimensions": orig_dims,
        "processed_dimensions": proc_dims,
        "metadata": metadata,
        "error": None,
    }
