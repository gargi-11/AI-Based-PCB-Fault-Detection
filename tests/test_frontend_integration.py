"""
Integration tests for the frontend dashboard components and end-to-end flow.
"""

import os
import unittest
from pathlib import Path
import cv2
import numpy as np

from app.app import find_sample_boards, get_services
from src.processing.pipeline import PCBInspectionPipeline
from src.services.analysis_service import AnalysisService
from src.services.storage_service import FirebaseStorageService
from src.agents.diagnostic_agent import DiagnosticAgent


class TestFrontendIntegration(unittest.TestCase):
    """Test suite verifying end-to-end integration for the dashboard."""

    def setUp(self):
        self.services = get_services()

    def test_services_initialization(self):
        """Verify that all backend services load and initialize correctly."""
        self.assertIsNotNone(self.services["firebase"])
        self.assertIsNotNone(self.services["storage"])
        self.assertIsNotNone(self.services["analysis"])
        self.assertIsNotNone(self.services["rag"])
        self.assertIsNotNone(self.services["agent"])
        self.assertIsNotNone(self.services["pipeline"])

    def test_find_sample_boards(self):
        """Verify that sample boards can be located in repository."""
        samples = find_sample_boards()
        self.assertIsInstance(samples, dict)
        if samples:
            first_sample = list(samples.values())[0]
            self.assertTrue(os.path.isfile(first_sample["test"]))
            self.assertTrue(os.path.isfile(first_sample["template"]))

    def test_end_to_end_pipeline_flow(self):
        """Verify the complete flow from image inspection to diagnostic report generation and persistence."""
        pipeline: PCBInspectionPipeline = self.services["pipeline"]
        analysis_service: AnalysisService = self.services["analysis"]
        storage_service: FirebaseStorageService = self.services["storage"]
        agent: DiagnosticAgent = self.services["agent"]

        # Create synthetic test and reference boards
        test_img = np.ones((640, 640, 3), dtype=np.uint8) * 40
        ref_img = np.ones((640, 640, 3), dtype=np.uint8) * 40

        # Draw a track on both
        cv2.line(ref_img, (100, 300), (500, 300), (0, 200, 100), 10)
        # Broken track on test image
        cv2.line(test_img, (100, 300), (280, 300), (0, 200, 100), 10)
        cv2.line(test_img, (340, 300), (500, 300), (0, 200, 100), 10)

        user_id = "test_user_frontend"
        aid = "ANA-TEST-E2E-001"

        # 1. Storage Upload
        upload_res = storage_service.upload_input_image(
            user_id=user_id, analysis_id=aid, file_source=test_img
        )
        self.assertIn("url", upload_res)

        # 2. Initialize Analysis
        analysis_doc = analysis_service.create_analysis(
            user_id=user_id,
            input_image_url=upload_res.get("url", "pcb_uploads/test.png"),
            analysis_id=aid,
        )
        self.assertEqual(analysis_doc["status"], "pending")

        # 3. CV Pipeline Inspection
        cv_res = pipeline.inspect_pcb(
            test_image_source=test_img,
            reference_image_source=ref_img,
            analysis_id=aid,
        )
        self.assertTrue(cv_res["success"])
        self.assertIn("faults", cv_res)
        self.assertIn("annotated_image", cv_res)

        # 4. Diagnostic Agent Reasoning (using offline / heuristic mode)
        faults = cv_res["faults"]
        reports = []
        for fault in faults:
            report = agent.diagnose(fault)
            reports.append(report)
            self.assertIn("detected_fault", report)
            self.assertIn("probable_causes", report)
            self.assertIn("recommended_action", report)
            self.assertIn("supporting_sources", report)

        # 5. Persist Final Analysis Results
        final_doc = analysis_service.save_analysis_results(
            analysis_id=aid,
            faults=faults,
            overall_result=cv_res.get("overall_result", "DEFECTIVE"),
            confidence=0.95,
            processing_time=0.15,
            diagnostic_report={"reports": reports},
        )
        self.assertEqual(final_doc["status"], "completed")

        # 6. Retrieve from Service
        retrieved = analysis_service.get_analysis(aid)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["analysis_id"], aid)


if __name__ == "__main__":
    unittest.main()
