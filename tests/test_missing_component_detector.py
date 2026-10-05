"""
Comprehensive unit test suite for Missing Component Detection:
- Structural Similarity (SSIM) map computation
- CV Baseline Reference-Differential Missing Component Detection
- False-positive rejection under illumination changes
- Standalone pad exposure detection
- Multi-panel diagnostic visualization generation
- Deep Learning detector interface & weights loading hook
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np

from src.processing.missing_component_detector import (
    CVMissingComponentDetector,
    DeepLearningMissingComponentDetector,
    compute_ssim_map,
    get_missing_component_detector,
)


class TestMissingComponentDetector(unittest.TestCase):
    """Test suite for missing component detection algorithms."""

    def setUp(self):
        """Create synthetic PCB test images."""
        self.test_dir = tempfile.mkdtemp(prefix="pcb_mc_test_")
        self.vis_dir = os.path.join(self.test_dir, "detected_components")
        self.h, self.w = 400, 500

        # 1. Golden Reference PCB
        self.ref_pcb = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.ref_pcb[:] = (34, 139, 34)  # Green soldermask

        # Populated IC Component (U1 at center 200, 150)
        cv2.rectangle(self.ref_pcb, (180, 130), (300, 230), (30, 30, 30), -1)
        cv2.putText(self.ref_pcb, "NE555", (200, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
        # Add IC pins
        for py in range(145, 220, 20):
            cv2.rectangle(self.ref_pcb, (165, py), (180, py + 8), (200, 200, 200), -1)
            cv2.rectangle(self.ref_pcb, (300, py), (315, py + 8), (200, 200, 200), -1)

        # Populated SMD Capacitor C1 (at 80, 280)
        cv2.rectangle(self.ref_pcb, (70, 270), (110, 295), (60, 90, 140), -1)
        cv2.rectangle(self.ref_pcb, (65, 270), (70, 295), (210, 210, 210), -1)
        cv2.rectangle(self.ref_pcb, (110, 270), (115, 295), (210, 210, 210), -1)

        # 2. Defective Test PCB (Missing IC U1 and Missing Capacitor C1)
        self.test_pcb = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.test_pcb[:] = (34, 139, 34)

        # In place of U1: blank soldermask with bare unpopulated pads
        for py in range(145, 220, 20):
            cv2.rectangle(self.test_pcb, (165, py), (180, py + 8), (190, 190, 190), -1)
            cv2.rectangle(self.test_pcb, (300, py), (315, py + 8), (190, 190, 190), -1)

        # In place of C1: exposed vacant solder pads with dark gap
        cv2.rectangle(self.test_pcb, (65, 270), (70, 295), (200, 200, 200), -1)
        cv2.rectangle(self.test_pcb, (110, 270), (115, 295), (200, 200, 200), -1)

        self.detector = CVMissingComponentDetector(
            min_component_area_px=50,
            visualizations_dir=self.vis_dir,
        )

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_01_ssim_computation(self):
        """Test structural similarity metrics between identical and differing images."""
        gray_ref = cv2.cvtColor(self.ref_pcb, cv2.COLOR_BGR2GRAY)
        gray_test = cv2.cvtColor(self.test_pcb, cv2.COLOR_BGR2GRAY)

        # Identical images must have SSIM close to 1.0
        ssim_same, _ = compute_ssim_map(gray_ref, gray_ref)
        self.assertAlmostEqual(ssim_same, 1.0, places=2)

        # Different images must have lower global SSIM than identical
        ssim_diff, dissim_map = compute_ssim_map(gray_ref, gray_test)
        self.assertLess(ssim_diff, ssim_same)
        self.assertEqual(dissim_map.shape, gray_ref.shape)

        # Local ROI of missing IC U1 (180, 130 to 300, 230) should have dramatic SSIM drop (< 0.60)
        roi_ref = gray_ref[130:230, 180:300]
        roi_test = gray_test[130:230, 180:300]
        local_ssim, _ = compute_ssim_map(roi_ref, roi_test)
        self.assertLess(local_ssim, 0.60)

    def test_02_detect_missing_components_reference_mode(self):
        """Test missing component detection using golden reference comparison."""
        faults = self.detector.detect_missing_components(
            test_image_bgr=self.test_pcb,
            reference_image_bgr=self.ref_pcb,
            analysis_id="ANA-MC-001",
        )

        self.assertIsInstance(faults, list)
        self.assertGreaterEqual(len(faults), 1)

        # Check required schema fields
        for f in faults:
            self.assertIn("component_id", f)
            self.assertIn("location", f)
            self.assertIn("bbox", f["location"])
            self.assertIn("confidence", f)
            self.assertIn("evidence", f)
            self.assertTrue(f["reference_present"])
            self.assertFalse(f["test_present"])
            self.assertIn("severity", f)
            self.assertEqual(f["fault_type"], "missing_component")

    def test_03_false_positive_rejection_under_lighting(self):
        """Test that global illumination change does not produce false missing components."""
        # Create test image by brightening the reference PCB slightly (+20 brightness)
        bright_test = np.clip(self.ref_pcb.astype(np.int16) + 20, 0, 255).astype(np.uint8)

        faults = self.detector.detect_missing_components(
            test_image_bgr=bright_test,
            reference_image_bgr=self.ref_pcb,
            analysis_id="ANA-BRIGHT",
        )
        # All components are present in bright_test -> should return 0 missing components
        self.assertEqual(len(faults), 0)

    def test_04_visualization_generator(self):
        """Test generation and disk persistence of multi-panel visualization."""
        faults = self.detector.detect_missing_components(
            test_image_bgr=self.test_pcb,
            reference_image_bgr=self.ref_pcb,
            analysis_id="ANA-VIS-TEST",
        )

        vis_img = self.detector.generate_visualization(
            test_image_bgr=self.test_pcb,
            reference_image_bgr=self.ref_pcb,
            detected_missing=faults,
            analysis_id="ANA-VIS-TEST",
        )

        self.assertIsNotNone(vis_img)
        expected_file = os.path.join(self.vis_dir, "missing_components_ANA-VIS-TEST.png")
        self.assertTrue(os.path.exists(expected_file))

    def test_05_deep_learning_detector_fallback(self):
        """Test DeepLearningMissingComponentDetector gracefully handles missing weights."""
        dl_detector = DeepLearningMissingComponentDetector(model_weights_path="non_existent_weights.pt")
        self.assertFalse(dl_detector.is_model_loaded)

        # Factory returns CV detector by default when no weights provided
        factory_detector = get_missing_component_detector()
        self.assertIsInstance(factory_detector, CVMissingComponentDetector)


if __name__ == "__main__":
    unittest.main()
