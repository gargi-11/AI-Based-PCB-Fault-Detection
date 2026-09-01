"""
Unit tests for PCB Image Preprocessing Module.
Uses synthetic test patterns to validate functionality without requiring real PCB images.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np

# Import module under test
from src.processing.preprocess import (
    enhance_pcb_illumination,
    preprocess_pcb_image,
    reduce_pcb_noise,
    resize_preserve_aspect,
)


class TestPCBImagePreprocessing(unittest.TestCase):
    """Test suite for preprocess_pcb_image and helper functions."""

    def setUp(self):
        """Create a temporary directory and synthetic test images."""
        self.test_dir = tempfile.mkdtemp(prefix="pcb_test_")

        # 1. Standard synthetic PCB-like image (400 height x 600 width x 3 channels)
        self.std_height, self.std_width = 400, 600
        self.std_image = np.zeros((self.std_height, self.std_width, 3), dtype=np.uint8)
        # Green PCB soldermask background (BGR: ~34, 139, 34)
        self.std_image[:] = (34, 139, 34)
        # Draw copper-colored traces (BGR: ~50, 180, 220)
        cv2.line(self.std_image, (50, 200), (550, 200), (50, 180, 220), 4)
        cv2.line(self.std_image, (200, 50), (200, 350), (50, 180, 220), 4)
        # Draw some rectangular solder pads (silver/gray)
        cv2.rectangle(self.std_image, (180, 180), (220, 220), (192, 192, 192), -1)
        cv2.rectangle(self.std_image, (380, 180), (420, 220), (192, 192, 192), -1)

        self.std_image_path = os.path.join(self.test_dir, "synthetic_pcb_std.png")
        cv2.imwrite(self.std_image_path, self.std_image)

        # 2. Large synthetic image (2000 height x 3000 width x 3 channels)
        self.large_height, self.large_width = 2000, 3000
        self.large_image = np.full((self.large_height, self.large_width, 3), 128, dtype=np.uint8)
        self.large_image_path = os.path.join(self.test_dir, "synthetic_pcb_large.png")
        cv2.imwrite(self.large_image_path, self.large_image)

        # 3. Corrupt/Invalid file
        self.corrupt_image_path = os.path.join(self.test_dir, "corrupt_image.png")
        with open(self.corrupt_image_path, "w") as f:
            f.write("NOT_A_VALID_IMAGE_FILE_DATA")

    def tearDown(self):
        """Clean up temporary test files."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_01_import_and_callable(self):
        """Verify that preprocess_pcb_image is imported and callable."""
        self.assertTrue(callable(preprocess_pcb_image))
        self.assertTrue(callable(resize_preserve_aspect))
        self.assertTrue(callable(enhance_pcb_illumination))
        self.assertTrue(callable(reduce_pcb_noise))

    def test_02_missing_file_handling(self):
        """Verify graceful error response when file does not exist."""
        non_existent_path = os.path.join(self.test_dir, "non_existent_pcb_file.png")
        result = preprocess_pcb_image(non_existent_path)

        self.assertIsInstance(result, dict)
        self.assertFalse(result["success"])
        self.assertIsNotNone(result["error"])
        self.assertIn("not found", result["error"].lower())
        self.assertIsNone(result["original_image"])
        self.assertIsNone(result["processed_image"])
        self.assertIsNone(result["grayscale_image"])

    def test_03_corrupt_file_handling(self):
        """Verify graceful error response when file is unreadable/corrupt."""
        result = preprocess_pcb_image(self.corrupt_image_path)

        self.assertIsInstance(result, dict)
        self.assertFalse(result["success"])
        self.assertIsNotNone(result["error"])
        self.assertIsNone(result["original_image"])

    def test_04_returned_dictionary_structure(self):
        """Verify all required dictionary keys and types are present on success."""
        result = preprocess_pcb_image(self.std_image_path)

        expected_keys = {
            "success",
            "original_image",
            "processed_image",
            "grayscale_image",
            "original_dimensions",
            "processed_dimensions",
            "metadata",
            "error",
        }
        self.assertTrue(expected_keys.issubset(result.keys()))
        self.assertTrue(result["success"])
        self.assertIsNone(result["error"])

        # Check image array types
        self.assertIsInstance(result["original_image"], np.ndarray)
        self.assertIsInstance(result["processed_image"], np.ndarray)
        self.assertIsInstance(result["grayscale_image"], np.ndarray)

        # Check dimensions tuples
        self.assertEqual(result["original_dimensions"], (self.std_height, self.std_width, 3))
        self.assertEqual(result["processed_dimensions"], (self.std_height, self.std_width, 3))

        # Check metadata
        metadata = result["metadata"]
        self.assertIsInstance(metadata, dict)
        self.assertIn("scale_factor", metadata)
        self.assertIn("operations_applied", metadata)
        self.assertEqual(metadata["scale_factor"], 1.0)
        self.assertFalse(metadata["is_resized"])

    def test_05_resizing_logic_large_image(self):
        """Verify that images exceeding max_dimension are resized preserving aspect ratio."""
        target_max_dim = 1500
        result = preprocess_pcb_image(self.large_image_path, max_dimension=target_max_dim)

        self.assertTrue(result["success"])
        self.assertEqual(result["original_dimensions"], (2000, 3000, 3))

        # Original aspect ratio = 3000 / 2000 = 1.5
        # Expected new dimensions: Width = 1500, Height = 1000
        proc_h, proc_w, proc_c = result["processed_dimensions"]
        self.assertEqual(proc_w, 1500)
        self.assertEqual(proc_h, 1000)
        self.assertEqual(proc_c, 3)

        self.assertTrue(result["metadata"]["is_resized"])
        self.assertAlmostEqual(result["metadata"]["scale_factor"], 0.5, places=3)
        self.assertEqual(result["processed_image"].shape, (1000, 1500, 3))
        self.assertEqual(result["grayscale_image"].shape, (1000, 1500))

    def test_06_grayscale_output(self):
        """Verify grayscale conversion properties."""
        result = preprocess_pcb_image(self.std_image_path)
        gray = result["grayscale_image"]

        self.assertTrue(result["success"])
        self.assertEqual(len(gray.shape), 2)  # Single channel (height, width)
        self.assertEqual(gray.shape, (self.std_height, self.std_width))
        self.assertEqual(gray.dtype, np.uint8)

    def test_07_optional_output_saving(self):
        """Verify saving processed image to an output path."""
        output_save_path = os.path.join(self.test_dir, "output", "processed_pcb.png")
        result = preprocess_pcb_image(self.std_image_path, output_path=output_save_path)

        self.assertTrue(result["success"])
        self.assertTrue(os.path.exists(output_save_path))
        self.assertTrue(result["metadata"].get("saved_successfully", False))

        # Verify saved file is a valid image on disk
        saved_img = cv2.imread(output_save_path)
        self.assertIsNotNone(saved_img)
        self.assertEqual(saved_img.shape, (self.std_height, self.std_width, 3))


if __name__ == "__main__":
    unittest.main()
