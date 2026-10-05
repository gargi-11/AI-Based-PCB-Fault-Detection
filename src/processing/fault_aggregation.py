"""
Fault Aggregation & Deduplication Module for PCB Fault Detection.

Aggregates component faults, track fractures, and differential anomalies.
Performs spatial Non-Maximum Suppression (NMS), determines overall PCB inspection
verdict, and outputs structured standardized detection payload.
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
import uuid
import numpy as np

logger = logging.getLogger(__name__)


def compute_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """
    Compute Intersection over Union (IoU) of two bounding boxes in [x, y, w, h] format.
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def deduplicate_faults(faults: List[Dict[str, Any]], iou_threshold: float = 0.35) -> List[Dict[str, Any]]:
    """
    Deduplicate overlapping bounding boxes using Non-Maximum Suppression (NMS).
    Keeps the fault with higher confidence.
    """
    if not faults:
        return []

    # Sort faults descending by confidence
    sorted_faults = sorted(faults, key=lambda f: f.get("confidence", 0.0), reverse=True)
    kept_faults: List[Dict[str, Any]] = []

    for candidate in sorted_faults:
        cand_bbox = candidate.get("location", {}).get("bbox")
        if not cand_bbox or len(cand_bbox) < 4:
            kept_faults.append(candidate)
            continue

        overlap = False
        for existing in kept_faults:
            ex_bbox = existing.get("location", {}).get("bbox")
            if ex_bbox and len(ex_bbox) >= 4:
                iou = compute_iou(cand_bbox, ex_bbox)
                if iou >= iou_threshold:
                    # Merge evidence points if relevant
                    cand_ev = candidate.get("evidence", [])
                    ex_ev = existing.get("evidence", [])
                    if isinstance(cand_ev, list) and isinstance(ex_ev, list):
                        for ev in cand_ev:
                            if ev not in ex_ev:
                                ex_ev.append(ev)
                    overlap = True
                    break

        if not overlap:
            kept_faults.append(candidate)

    return kept_faults


def aggregate_faults(
    component_faults: List[Dict[str, Any]],
    track_faults: List[Dict[str, Any]],
    diff_anomalies: Optional[List[Dict[str, Any]]] = None,
    reference_used: bool = False,
    analysis_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Aggregate all defect findings into a standardized inspection result.

    Args:
        component_faults: List of missing/misplaced component faults.
        track_faults: List of broken copper track faults.
        diff_anomalies: Optional raw differential anomaly regions.
        reference_used: Boolean indicating if golden reference comparison was used.
        analysis_id: Optional session identifier.

    Returns:
        Standardized detection payload dict.
    """
    aid = analysis_id or f"ANA-{uuid.uuid4().hex[:8].upper()}"
    raw_faults = list(component_faults) + list(track_faults)

    # 1. Deduplicate overlapping fault bounding boxes
    final_faults = deduplicate_faults(raw_faults)

    # 2. Renumber faults with clean IDs
    for idx, f in enumerate(final_faults, start=1):
        f["fault_id"] = f"FLT-{idx:03d}"

    # 3. Compute Defect Breakdown
    breakdown: Dict[str, int] = {}
    for f in final_faults:
        ftype = f.get("fault_type", "other")
        breakdown[ftype] = breakdown.get(ftype, 0) + 1

    total_faults = len(final_faults)

    # 4. Determine Overall Verdict & Severity
    has_critical = any(
        f.get("severity") in ("critical_open", "critical", "high") for f in final_faults
    )

    if total_faults == 0:
        overall_result = "PASS"
        summary = (
            "Inspection Complete: PCB is fully continuous with all expected components populated. "
            "No physical defects detected."
        )
        avg_confidence = 0.98 if reference_used else 0.90
    else:
        overall_result = "CRITICAL_FAULT" if has_critical else "DEFECTIVE"
        parts = []
        if breakdown.get("missing_component"):
            parts.append(f"{breakdown['missing_component']} missing component(s)")
        if breakdown.get("broken_copper_track"):
            parts.append(f"{breakdown['broken_copper_track']} broken copper track(s)")
        for k, v in breakdown.items():
            if k not in ("missing_component", "broken_copper_track"):
                parts.append(f"{v} {k} defect(s)")

        mode_str = "golden reference differential analysis" if reference_used else "standalone computer vision inspection"
        summary = f"Inspection Complete ({mode_str}): Identified {', '.join(parts)} requiring rework."
        avg_confidence = float(np.mean([f.get("confidence", 0.8) for f in final_faults]))

    return {
        "analysis_id": aid,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_result": overall_result,
        "confidence": round(avg_confidence, 4),
        "total_defects": total_faults,
        "defect_breakdown": breakdown,
        "summary": summary,
        "reference_used": bool(reference_used),
        "faults": final_faults,
    }
