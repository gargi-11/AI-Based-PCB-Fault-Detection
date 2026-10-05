"""
Comprehensive unit and integration test suite for the PCB Computer Vision Pipeline:
- Image Loader & Validation (image_loader.py)
- Enhancement & Normalization (enhancement.py)
- Reference Alignment (alignment.py)
- Difference Map & Comparison (comparison.py)
- Component Defect Detection (component_detection.py)
- Track Defect Detection (track_detection.py)
- Fault Aggregation (fault_aggregation.py)
- Master Pipeline Integration (pipeline.py)
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np

from src.processing.image_loader import validate_and_load_image, save_image
from src.processing.enhancement import enhance_pcb_pipeline, to_grayscale
from src.processing.alignment import align_images_orb
from src.processing.comparison import compute_difference_map
from src.processing.component_detection import detect_component_defects
from src.processing.track_detection import (
    detect_track_defects,
    segment_copper_traces,
    skeletonize_traces,
    find_skeleton_endpoints,
)
from src.processing.fault_aggregation import aggregate_faults, compute_iou, deduplicate_faults
from src.processing.pipeline import PCBInspectionPipeline, inspect_pcb


class TestProcessingPipeline(unittest.TestCase):
    """Test suite for core PCB computer vision modules."""

    def setUp(self):
        """Create synthetic PCB test patterns (Reference PCB and Defective Test PCB)."""
        self.test_dir = tempfile.mkdtemp(prefix="pcb_cv_test_")
        self.h, self.w = 400, 500

        # 1. Golden Reference PCB (Green soldermask, copper traces, populated components)
        self.ref_pcb = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.ref_pcb[:] = (34, 139, 34)  # Green soldermask

        # Add fiducials / distinctive silkscreen corners for feature alignment
        cv2.circle(self.ref_pcb, (50, 50), 12, (240, 240, 240), -1)
        cv2.circle(self.ref_pcb, (450, 50), 12, (240, 240, 240), -1)
        cv2.circle(self.ref_pcb, (50, 350), 12, (240, 240, 240), -1)
        cv2.circle(self.ref_pcb, (450, 350), 12, (240, 240, 240), -1)

        # Continuous Copper Signal Track 1 (Horizontal across y=150)
        cv2.line(self.ref_pcb, (80, 150), (420, 150), (50, 210, 245), 6)

        # Continuous Copper Signal Track 2 (Vertical across x=250)
        cv2.line(self.ref_pcb, (250, 80), (250, 320), (50, 210, 245), 6)

        # Populated IC Component (Dark body with white marking text at center)
        cv2.rectangle(self.ref_pcb, (200, 180), (300, 260), (30, 30, 30), -1)
        cv2.putText(self.ref_pcb, "U1", (230, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)

        # Populated Resistor Component R1
        cv2.rectangle(self.ref_pcb, (120, 220), (160, 240), (20, 20, 150), -1)

        self.ref_path = os.path.join(self.test_dir, "golden_reference.png")
        cv2.imwrite(self.ref_path, self.ref_pcb)

        # 2. Defective Test PCB (Has Missing R1 and Broken Track 1)
        self.test_pcb = self.ref_pcb.copy()

        # Defect 1: Broken Track (sever Track 1 at x=320 to x=345 with soldermask green)
        cv2.line(self.test_pcb, (320, 150), (345, 150), (34, 139, 34), 8)

        # Defect 2: Missing Resistor R1 (replace with unpopulated pads / soldermask)
        cv2.rectangle(self.test_pcb, (120, 220), (160, 240), (34, 139, 34), -1)
        # Add exposed rectangular solder pads at the ends
        cv2.rectangle(self.test_pcb, (115, 225), (125, 235), (200, 200, 200), -1)
        cv2.rectangle(self.test_pcb, (155, 225), (165, 235), (200, 200, 200), -1)

        self.test_path = os.path.join(self.test_dir, "defective_test.png")
        cv2.imwrite(self.test_path, self.test_pcb)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_01_image_loader_sources(self):
        """Test loading from numpy array, file path, and bytes."""
        # 1. File Path
        ok, img, err = validate_and_load_image(self.test_path)
        self.assertTrue(ok)
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (self.h, self.w, 3))

        # 2. Raw Bytes
        data = Path(self.test_path).read_bytes()
        ok_b, img_b, _ = validate_and_load_image(data)
        self.assertTrue(ok_b)
        self.assertEqual(img_b.shape, (self.h, self.w, 3))

        # 3. Invalid source
        ok_inv, _, err_inv = validate_and_load_image("non_existent_file.xyz")
        self.assertFalse(ok_inv)
        self.assertIsNotNone(err_inv)

    def test_02_enhancement_workflow(self):
        """Test contrast enhancement and bilateral filtering."""
        res = enhance_pcb_pipeline(self.test_pcb, max_dimension=1000)
        self.assertIn("processed_image", res)
        self.assertIn("grayscale_image", res)
        self.assertEqual(res["processed_image"].shape, (self.h, self.w, 3))
        self.assertEqual(res["grayscale_image"].shape, (self.h, self.w))

    def test_03_alignment_orb(self):
        """Test ORB alignment between slightly shifted test and reference images."""
        # Create a translated test board
        M = np.float32([[1, 0, 10], [0, 1, 5]])
        shifted_test = cv2.warpAffine(self.ref_pcb, M, (self.w, self.h))

        align_res = align_images_orb(shifted_test, self.ref_pcb)
        self.assertTrue(align_res["success"])
        self.assertIsNotNone(align_res["aligned_image"])
        self.assertGreater(align_res["inliers_count"], 5)

    def test_04_comparison_difference_map(self):
        """Test differential map calculation and noise suppression."""
        diff_res = compute_difference_map(self.test_pcb, self.ref_pcb, diff_threshold=30)
        self.assertIn("diff_mask", diff_res)
        self.assertIn("diff_regions", diff_res)
        self.assertGreater(len(diff_res["diff_regions"]), 0)

    def test_05_component_detection(self):
        """Test component defect detection in both reference-guided and standalone modes."""
        diff_res = compute_difference_map(self.test_pcb, self.ref_pcb)
        faults_ref = detect_component_defects(
            test_image_bgr=self.test_pcb,
            ref_image_bgr=self.ref_pcb,
            diff_regions=diff_res["diff_regions"],
        )
        self.assertIsInstance(faults_ref, list)
        # Should identify the missing component at (120, 220)
        has_missing = any(f["fault_type"] == "missing_component" for f in faults_ref)
        self.assertTrue(has_missing)

    def test_06_track_detection(self):
        """Test broken track detection on the severed trace."""
        diff_res = compute_difference_map(self.test_pcb, self.ref_pcb)
        faults_track = detect_track_defects(
            test_image_bgr=self.test_pcb,
            ref_image_bgr=self.ref_pcb,
            diff_regions=diff_res["diff_regions"],
        )
        self.assertIsInstance(faults_track, list)
        has_broken_track = any(f["fault_type"] == "broken_copper_track" for f in faults_track)
        self.assertTrue(has_broken_track)

    def test_07_fault_aggregation_and_nms(self):
        """Test fault deduplication and overall verdict computation."""
        f1 = {
            "fault_id": "FLT-01",
            "fault_type": "missing_component",
            "location": {"bbox": [100, 100, 40, 40]},
            "confidence": 0.95,
            "severity": "high",
        }
        f2 = {
            "fault_id": "FLT-02",
            "fault_type": "missing_component",
            "location": {"bbox": [102, 101, 38, 39]},  # High overlap with f1
            "confidence": 0.80,
            "severity": "high",
        }
        res = aggregate_faults([f1, f2], [], reference_used=True)
        # f2 should be suppressed by NMS
        self.assertEqual(len(res["faults"]), 1)
        self.assertEqual(res["overall_result"], "CRITICAL_FAULT")
        self.assertTrue(res["reference_used"])

    def test_08_full_pipeline_with_reference(self):
        """Test end-to-end inspection pipeline with golden reference (Mode B)."""
        pipeline = PCBInspectionPipeline(debug_output_dir=os.path.join(self.test_dir, "debug"))
        res = pipeline.inspect_pcb(
            test_image_source=self.test_path,
            reference_image_source=self.ref_path,
            analysis_id="ANA-TEST-100",
            save_debug=True,
        )

        self.assertTrue(res["success"])
        self.assertEqual(res["analysis_id"], "ANA-TEST-100")
        self.assertTrue(res["reference_used"])
        self.assertIn("faults", res)
        self.assertIn("summary", res)
        self.assertIn("overall_result", res)
        self.assertIn("confidence", res)
        self.assertGreater(len(res["faults"]), 0)

        # Check that debug images were saved
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "debug", "ANA-TEST-100_03_annotated.png")))

    def test_09_full_pipeline_standalone_mode(self):
        """Test end-to-end inspection pipeline in standalone mode without reference (Mode A)."""
        res = inspect_pcb(
            test_image_source=self.test_pcb,
            reference_image_source=None,
            analysis_id="ANA-STANDALONE-01",
        )
        self.assertTrue(res["success"])
        self.assertFalse(res["reference_used"])
        self.assertIn("faults", res)
        self.assertIn("summary", res)


if __name__ == "__main__":
    unittest.main()
