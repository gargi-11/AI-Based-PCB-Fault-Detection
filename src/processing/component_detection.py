"""
PCB Component Defect Detection Module.

Authoritative Computer Vision & Differential Analysis for Missing Components
and Component Placement Anomalies. Integrates with CVMissingComponentDetector.
"""

import logging
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from src.processing.missing_component_detector import (
    CVMissingComponentDetector,
    BaseMissingComponentDetector,
    get_missing_component_detector,
)

logger = logging.getLogger(__name__)


def detect_component_defects(
    test_image_bgr: np.ndarray,
    ref_image_bgr: Optional[np.ndarray] = None,
    diff_regions: Optional[List[Dict[str, Any]]] = None,
    min_component_area_px: int = 60,
    analysis_id: Optional[str] = None,
    generate_vis: bool = True,
) -> List[Dict[str, Any]]:
    """
    Detect missing or misplaced components on the PCB using Computer Vision.

    Args:
        test_image_bgr: Preprocessed test PCB image.
        ref_image_bgr: Optional preprocessed golden reference PCB image.
        diff_regions: Optional pre-computed differential anomaly regions.
        min_component_area_px: Minimum pixel area threshold.
        analysis_id: Optional inspection session ID.
        generate_vis: Whether to generate and save visual comparison to results/detected_components/.

    Returns:
        List of detected component fault dictionaries with schema:
        - component_id: str
        - location: dict (bbox: [x, y, w, h], area_px: int)
        - confidence: float
        - evidence: list[str]
        - reference_present: bool
        - test_present: bool
        - severity: str
        - fault_id: str
        - fault_type: "missing_component"
    """
    detector = CVMissingComponentDetector(
        min_component_area_px=min_component_area_px,
    )

    detected = detector.detect_missing_components(
        test_image_bgr=test_image_bgr,
        reference_image_bgr=ref_image_bgr,
        analysis_id=analysis_id,
    )

    # Generate diagnostic visualization if requested
    if generate_vis and detected:
        detector.generate_visualization(
            test_image_bgr=test_image_bgr,
            reference_image_bgr=ref_image_bgr,
            detected_missing=detected,
            analysis_id=analysis_id,
        )

    return detected
