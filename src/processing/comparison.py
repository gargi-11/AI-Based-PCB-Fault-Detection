"""
PCB Differential Analysis & Golden Board Comparison Module.

Computes photometric and structural differences between aligned test PCB
and golden reference PCB, suppressing minor noise and isolating defect candidates.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def compute_difference_map(
    aligned_test_bgr: np.ndarray,
    ref_image_bgr: np.ndarray,
    diff_threshold: int = 35,
    min_area_px: int = 40,
    blur_ksize: int = 5,
) -> Dict[str, Any]:
    """
    Compute differential anomaly map between aligned test PCB and reference template.

    Args:
        aligned_test_bgr: Geometrically aligned test PCB image (BGR ndarray).
        ref_image_bgr: Reference template PCB image (BGR ndarray).
        diff_threshold: Pixel intensity threshold for considering a diff significant (0-255).
        min_area_px: Minimum contiguous pixel area to classify as a valid anomaly region.
        blur_ksize: Kernel size for pre-smoothing to suppress sub-pixel alignment jitter.

    Returns:
        Dict containing:
        - "diff_gray": np.ndarray (absolute difference grayscale)
        - "diff_mask": np.ndarray (cleaned binary anomaly mask)
        - "diff_heatmap": np.ndarray (colorized difference visualization)
        - "diff_regions": List[Dict] (bounding boxes and metrics for anomaly clusters)
        - "total_diff_area_px": int
        - "diff_ratio": float (percentage of PCB surface area with anomalies)
    """
    if aligned_test_bgr is None or ref_image_bgr is None:
        raise ValueError("aligned_test_bgr and ref_image_bgr cannot be None.")

    # Ensure identical dimensions
    h, w = ref_image_bgr.shape[:2]
    if aligned_test_bgr.shape[:2] != (h, w):
        aligned_test_bgr = cv2.resize(aligned_test_bgr, (w, h), interpolation=cv2.INTER_AREA)

    # 1. Convert to grayscale and apply mild Gaussian blur to suppress subpixel jitter
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    test_gray = cv2.cvtColor(aligned_test_bgr, cv2.COLOR_BGR2GRAY) if aligned_test_bgr.ndim == 3 else aligned_test_bgr
    ref_gray = cv2.cvtColor(ref_image_bgr, cv2.COLOR_BGR2GRAY) if ref_image_bgr.ndim == 3 else ref_image_bgr

    test_blur = cv2.GaussianBlur(test_gray, (k, k), 0)
    ref_blur = cv2.GaussianBlur(ref_gray, (k, k), 0)

    # 2. Compute absolute photometric difference
    diff_gray = cv2.absdiff(ref_blur, test_blur)

    # 3. Binary thresholding (fixed threshold + adaptive floor)
    _, raw_mask = cv2.threshold(diff_gray, diff_threshold, 255, cv2.THRESH_BINARY)

    # 4. Morphological noise suppression
    # Opening (erodes tiny specks / sensor noise) -> Closing (fills internal voids)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # 5. Extract Connected Components & Contours
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    diff_regions: List[Dict[str, Any]] = []
    total_anomaly_area = 0

    filtered_mask = np.zeros_like(clean_mask)

    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)
        # Extract region statistics
        roi_diff = diff_gray[y : y + ch, x : x + cw]
        mean_delta = float(np.mean(roi_diff))

        # Check reference vs test brightness in this region
        ref_roi = ref_gray[y : y + ch, x : x + cw]
        test_roi = test_gray[y : y + ch, x : x + cw]
        ref_mean = float(np.mean(ref_roi))
        test_mean = float(np.mean(test_roi))

        cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)
        total_anomaly_area += int(area)

        diff_regions.append(
            {
                "region_id": f"DIFF-REG-{idx + 1:03d}",
                "bbox": [int(x), int(y), int(cw), int(ch)],
                "area_px": int(area),
                "mean_intensity_delta": round(mean_delta, 2),
                "ref_mean_intensity": round(ref_mean, 2),
                "test_mean_intensity": round(test_mean, 2),
                "intensity_direction": "darker_in_test" if test_mean < ref_mean else "brighter_in_test",
            }
        )

    # 6. Generate Colorized Difference Heatmap
    diff_heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)

    total_pcb_pixels = h * w
    diff_ratio = (total_anomaly_area / total_pcb_pixels) if total_pcb_pixels > 0 else 0.0

    return {
        "diff_gray": diff_gray,
        "diff_mask": filtered_mask,
        "diff_heatmap": diff_heatmap,
        "diff_regions": diff_regions,
        "total_diff_area_px": total_anomaly_area,
        "diff_ratio": round(float(diff_ratio), 5),
    }
