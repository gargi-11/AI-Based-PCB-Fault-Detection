"""
PCB Copper Track Defect Detection Module.

Authoritative Computer Vision & Graph Skeletonization for detecting broken,
severed, and damaged copper tracks. Integrates with CopperTrackDefectDetector.
"""

import logging
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from src.processing.track_defect_detector import (
    CopperTrackDefectDetector,
    segment_conductive_traces as segment_copper_traces,
    skeletonize_medial_axis as skeletonize_traces,
    extract_skeleton_endpoints as find_skeleton_endpoints,
)

logger = logging.getLogger(__name__)


def detect_track_defects(
    test_image_bgr: np.ndarray,
    ref_image_bgr: Optional[np.ndarray] = None,
    diff_regions: Optional[List[Dict[str, Any]]] = None,
    max_break_gap_px: int = 40,
    min_break_gap_px: int = 3,
    analysis_id: Optional[str] = None,
    generate_vis: bool = True,
) -> List[Dict[str, Any]]:
    """
    Detect broken or damaged copper tracks on the PCB using Computer Vision.

    Args:
        test_image_bgr: Preprocessed test PCB image.
        ref_image_bgr: Optional preprocessed golden reference PCB image.
        diff_regions: Optional differential anomaly regions.
        max_break_gap_px: Maximum distance between broken track endpoints.
        min_break_gap_px: Minimum gap distance to consider a physical fracture.
        analysis_id: Optional inspection session ID.
        generate_vis: Whether to generate and save visual comparison to results/detected_tracks/.

    Returns:
        List of detected track fault records with schema:
        - fault_type: "broken_copper_track" | "missing_track_segment" | "track_discontinuity"
        - location: dict (bbox: [x, y, w, h], gap_distance_px, endpoints)
        - confidence: float
        - severity: "critical_open" | "high" | "moderate"
        - evidence: list[str]
        - reference_difference: dict
    """
    detector = CopperTrackDefectDetector(
        min_break_gap_px=min_break_gap_px,
        max_break_gap_px=max_break_gap_px,
    )

    detected = detector.detect_track_defects(
        test_image_bgr=test_image_bgr,
        ref_image_bgr=ref_image_bgr,
        diff_regions=diff_regions,
        analysis_id=analysis_id,
    )

    if generate_vis and detected:
        detector.generate_visualization(
            test_image_bgr=test_image_bgr,
            reference_image_bgr=ref_image_bgr,
            detected_faults=detected,
            analysis_id=analysis_id,
        )

    return detected
