"""
Comprehensive unit test suite for Copper Track Defect Detection:
- Conductive trace segmentation from soldermask
- Medial axis topological skeletonization
- Skeleton graph endpoint extraction
- Broken track and gap discontinuity localization
- Golden reference differential track volume comparison
- Multi-panel diagnostic visualization generation
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np

from src.processing.track_defect_detector import (
    CopperTrackDefectDetector,
    segment_conductive_traces,
    skeletonize_medial_axis,
    extract_skeleton_endpoints,
)


class TestTrackDefectDetector(unittest.TestCase):
    """Test suite for copper track fault detection algorithms."""

    def setUp(self):
        """Create synthetic PCB track images."""
        self.test_dir = tempfile.mkdtemp(prefix="pcb_track_test_")
        self.vis_dir = os.path.join(self.test_dir, "detected_tracks")
        self.h, self.w = 400, 500

        # 1. Golden Reference PCB (Continuous Copper Traces on Green Soldermask)
        self.ref_pcb = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.ref_pcb[:] = (34, 139, 34)  # Green soldermask

        # Track 1: Continuous Horizontal Copper Rail (y=120, x=50 to 450)
        cv2.line(self.ref_pcb, (50, 120), (450, 120), (40, 200, 245), 6)

        # Track 2: Continuous Vertical Copper Bus (x=300, y=80 to 350)
        cv2.line(self.ref_pcb, (300, 80), (300, 350), (40, 200, 245), 6)

        # 2. Defective Test PCB (Has Track 1 severed at x=220-240, and missing segment on Track 2)
        self.test_pcb = self.ref_pcb.copy()

        # Sever Track 1 with a 20px gap (cut with green soldermask)
        cv2.line(self.test_pcb, (220, 120), (240, 120), (34, 139, 34), 8)

        self.detector = CopperTrackDefectDetector(
            min_break_gap_px=3,
            max_break_gap_px=40,
            visualizations_dir=self.vis_dir,
        )

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_01_trace_segmentation(self):
        """Test segmentation of conductive copper lines from soldermask substrate."""
        mask = segment_conductive_traces(self.ref_pcb)
        self.assertEqual(mask.shape, (self.h, self.w))
        self.assertEqual(mask.dtype, np.uint8)
        # Verify copper line pixels are active
        self.assertGreater(np.count_nonzero(mask[120, 60:440]), 200)

    def test_02_skeletonization_and_endpoints(self):
        """Test 1-pixel skeletonization and endpoint detection on a severed line."""
        # Create a simple severed horizontal line
        binary_img = np.zeros((100, 200), dtype=np.uint8)
        binary_img[50, 20:80] = 255   # Left segment
        binary_img[50, 100:160] = 255 # Right segment (20px gap between 80 and 100)

        skel = skeletonize_medial_axis(binary_img)
        self.assertEqual(skel.shape, (100, 200))

        endpoints = extract_skeleton_endpoints(skel)
        # The two disconnected line segments should yield 4 endpoints
        self.assertEqual(len(endpoints), 4)

    def test_03_detect_track_defects_reference_mode(self):
        """Test track defect detection using golden reference comparison."""
        faults = self.detector.detect_track_defects(
            test_image_bgr=self.test_pcb,
            ref_image_bgr=self.ref_pcb,
            analysis_id="ANA-TRK-001",
        )

        self.assertIsInstance(faults, list)
        self.assertGreaterEqual(len(faults), 1)

        # Check required schema fields
        for f in faults:
            self.assertIn("fault_type", f)
            self.assertIn("location", f)
            self.assertIn("bbox", f["location"])
            self.assertIn("confidence", f)
            self.assertIn("severity", f)
            self.assertIn("evidence", f)
            self.assertIn("reference_difference", f)
            self.assertIn(f["fault_type"], ("broken_copper_track", "missing_track_segment", "track_discontinuity"))

    def test_04_detect_track_defects_standalone_mode(self):
        """Test topological skeleton gap detection without a reference image."""
        faults = self.detector.detect_track_defects(
            test_image_bgr=self.test_pcb,
            ref_image_bgr=None,
            analysis_id="ANA-TRK-STANDALONE",
        )

        self.assertIsInstance(faults, list)
        self.assertGreaterEqual(len(faults), 1)
        # Should identify the gap on Track 1
        break_fault = next((f for f in faults if f["fault_type"] == "broken_copper_track"), None)
        self.assertIsNotNone(break_fault)
        self.assertIn("endpoints", break_fault["location"])

    def test_05_visualization_generation(self):
        """Test generation and persistence of track defect visualization composite."""
        faults = self.detector.detect_track_defects(
            test_image_bgr=self.test_pcb,
            ref_image_bgr=self.ref_pcb,
            analysis_id="ANA-TRK-VIS",
        )

        vis_img = self.detector.generate_visualization(
            test_image_bgr=self.test_pcb,
            reference_image_bgr=self.ref_pcb,
            detected_faults=faults,
            analysis_id="ANA-TRK-VIS",
        )

        self.assertIsNotNone(vis_img)
        expected_file = os.path.join(self.vis_dir, "track_defects_ANA-TRK-VIS.png")
        self.assertTrue(os.path.exists(expected_file))


if __name__ == "__main__":
    unittest.main()
