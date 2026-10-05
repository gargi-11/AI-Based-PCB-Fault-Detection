"""
Master PCB Image Inspection Pipeline.

Coordinates the full end-to-end computer vision analysis:
Input PCB Image -> Validation -> Resize -> Denoising -> Contrast Enhancement ->
Reference Alignment -> Comparison -> Component Analysis -> Track Analysis ->
Fault Aggregation -> Structured Output & Visual Overlay.
"""

from datetime import datetime, timezone
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from src.processing.image_loader import validate_and_load_image, save_image
from src.processing.enhancement import enhance_pcb_pipeline
from src.processing.alignment import align_images_orb
from src.processing.comparison import compute_difference_map
from src.processing.component_detection import detect_component_defects
from src.processing.track_detection import detect_track_defects
from src.processing.fault_aggregation import aggregate_faults

logger = logging.getLogger(__name__)


class PCBInspectionPipeline:
    """
    Executes end-to-end computer vision inspection for PCB photographs,
    supporting both standalone mode and golden-reference differential comparison mode.
    """

    def __init__(self, debug_output_dir: Optional[str] = None):
        self.debug_dir = Path(
            debug_output_dir
            or (Path(__file__).resolve().parent.parent.parent / "results" / "debug")
        )
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    def draw_defect_annotations(
        self,
        base_image_bgr: np.ndarray,
        faults: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Draw clean visual bounding boxes, confidence tags, and defect labels on the image.
        """
        annotated = base_image_bgr.copy()

        # Color palette (BGR):
        # Missing component: Red (0, 0, 230)
        # Broken track: Orange/Amber (0, 140, 255)
        # Other: Magenta (200, 0, 200)
        color_map = {
            "missing_component": (0, 0, 230),
            "broken_copper_track": (0, 140, 255),
            "solder_bridge": (255, 100, 0),
        }

        for fault in faults:
            loc = fault.get("location", {})
            bbox = loc.get("bbox")
            if not bbox or len(bbox) < 4:
                continue

            x, y, w, h = [int(v) for v in bbox[:4]]
            ftype = fault.get("fault_type", "defect")
            fid = fault.get("fault_id", "FLT")
            conf = fault.get("confidence", 0.0)
            color = color_map.get(ftype, (0, 0, 255))

            # Draw outer rectangle
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Draw corner markers
            corner_len = min(12, max(4, min(w, h) // 3))
            cv2.line(annotated, (x, y), (x + corner_len, y), color, 3)
            cv2.line(annotated, (x, y), (x, y + corner_len), color, 3)
            cv2.line(annotated, (x + w, y), (x + w - corner_len, y), color, 3)
            cv2.line(annotated, (x + w, y), (x + w, y + corner_len), color, 3)
            cv2.line(annotated, (x, y + h), (x + corner_len, y + h), color, 3)
            cv2.line(annotated, (x, y + h), (x, y + h - corner_len), color, 3)
            cv2.line(annotated, (x + w, y + h), (x + w - corner_len, y + h), color, 3)
            cv2.line(annotated, (x + w, y + h), (x + w, y + h - corner_len), color, 3)

            # Label banner
            label = f"{fid}: {ftype.replace('_', ' ').title()} ({int(conf * 100)}%)"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

            label_y = max(th + 5, y - 5)
            cv2.rectangle(
                annotated,
                (x, label_y - th - 3),
                (x + tw + 6, label_y + baseline),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x + 3, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated

    def inspect_pcb(
        self,
        test_image_source: Union[str, Path, bytes, np.ndarray, Image.Image],
        reference_image_source: Optional[Union[str, Path, bytes, np.ndarray, Image.Image]] = None,
        analysis_id: Optional[str] = None,
        save_debug: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute comprehensive PCB fault detection inspection.

        Args:
            test_image_source: Input test PCB image.
            reference_image_source: Optional golden reference PCB template.
            analysis_id: Optional unique inspection ID.
            save_debug: If True, writes debug visualization images to results/debug/.

        Returns:
            Structured dictionary matching requirements.
        """
        start_time = time.time()

        # 1. Validation & Loading
        valid_test, raw_test, err_test = validate_and_load_image(test_image_source)
        if not valid_test or raw_test is None:
            return {
                "analysis_id": analysis_id,
                "success": False,
                "error": f"Invalid test image: {err_test}",
                "faults": [],
                "summary": f"Validation failed: {err_test}",
                "reference_used": False,
                "overall_result": "FAILED",
                "confidence": 0.0,
            }

        # 2. Preprocess & Enhance Test Image
        test_prep = enhance_pcb_pipeline(raw_test)
        proc_test = test_prep["processed_image"]

        # 3. Check for Reference / Golden Board
        reference_used = False
        aligned_test = proc_test
        proc_ref = None
        diff_regions = []
        diff_heatmap = None
        alignment_meta = {}

        if reference_image_source is not None:
            valid_ref, raw_ref, _ = validate_and_load_image(reference_image_source)
            if valid_ref and raw_ref is not None:
                ref_prep = enhance_pcb_pipeline(raw_ref)
                proc_ref = ref_prep["processed_image"]

                # Align test image to reference space
                align_res = align_images_orb(proc_test, proc_ref)
                if align_res["success"]:
                    aligned_test = align_res["aligned_image"]
                    reference_used = True
                    alignment_meta = {
                        "aligned": True,
                        "inliers": align_res["inliers_count"],
                        "confidence": align_res["alignment_confidence"],
                    }
                else:
                    logger.warning("Alignment fell back; continuing with direct resolution matching.")
                    aligned_test = align_res["aligned_image"]
                    reference_used = False
                    alignment_meta = {"aligned": False, "reason": align_res.get("error")}

                # Compute difference map
                if proc_ref is not None:
                    diff_res = compute_difference_map(aligned_test, proc_ref)
                    diff_regions = diff_res["diff_regions"]
                    diff_heatmap = diff_res["diff_heatmap"]

        # 4. Component Defect Analysis
        component_faults = detect_component_defects(
            test_image_bgr=aligned_test,
            ref_image_bgr=proc_ref if reference_used else None,
            diff_regions=diff_regions if reference_used else None,
        )

        # 5. Copper Track Defect Analysis
        track_faults = detect_track_defects(
            test_image_bgr=aligned_test,
            ref_image_bgr=proc_ref if reference_used else None,
            diff_regions=diff_regions if reference_used else None,
        )

        # 6. Fault Aggregation & Deduplication
        result = aggregate_faults(
            component_faults=component_faults,
            track_faults=track_faults,
            diff_anomalies=diff_regions,
            reference_used=reference_used,
            analysis_id=analysis_id,
        )

        # 7. Render Defect Overlay
        annotated_image = self.draw_defect_annotations(aligned_test, result["faults"])
        elapsed = round(time.time() - start_time, 3)

        result["success"] = True
        result["processing_time_seconds"] = elapsed
        result["annotated_image"] = annotated_image
        result["processed_image"] = aligned_test
        result["diff_heatmap"] = diff_heatmap
        result["metadata"] = {
            "test_dimensions": raw_test.shape,
            "processed_dimensions": aligned_test.shape,
            "scale_factor": test_prep["scale_factor"],
            "alignment": alignment_meta,
        }

        # 8. Save debug visualizations if requested
        if save_debug:
            aid = result["analysis_id"]
            save_image(raw_test, self.debug_dir / f"{aid}_01_raw.png")
            save_image(aligned_test, self.debug_dir / f"{aid}_02_preprocessed.png")
            save_image(annotated_image, self.debug_dir / f"{aid}_03_annotated.png")
            if diff_heatmap is not None:
                save_image(diff_heatmap, self.debug_dir / f"{aid}_04_diff_heatmap.png")

        return result


def inspect_pcb(
    test_image_source: Union[str, Path, bytes, np.ndarray, Image.Image],
    reference_image_source: Optional[Union[str, Path, bytes, np.ndarray, Image.Image]] = None,
    analysis_id: Optional[str] = None,
    save_debug: bool = False,
) -> Dict[str, Any]:
    """Convenience function to run the full inspection pipeline."""
    pipeline = PCBInspectionPipeline()
    return pipeline.inspect_pcb(
        test_image_source=test_image_source,
        reference_image_source=reference_image_source,
        analysis_id=analysis_id,
        save_debug=save_debug,
    )
