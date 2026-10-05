"""
Comprehensive unit and integration test suite for Firebase Backend Services:
- Firebase Initialization (firebase_service.py)
- User Authentication (auth_service.py)
- Cloud/Local Image Storage (storage_service.py)
- Firestore Analysis Documents & Faults (analysis_service.py)
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np

from src.services.firebase_service import FirebaseConfig, FirebaseService
from src.services.auth_service import FirebaseAuthService, UserProfile
from src.services.storage_service import FirebaseStorageService
from src.services.analysis_service import (
    AnalysisService,
    FaultRecord,
    STATUS_PENDING,
    STATUS_PROCESSING,
    STATUS_COMPLETED,
    RESULT_DEFECTIVE,
    RESULT_PASS,
)


class TestFirebaseServices(unittest.TestCase):
    """Test suite for Firebase Backend modules."""

    def setUp(self):
        """Create isolated temporary test directories for storage and database."""
        self.test_dir = tempfile.mkdtemp(prefix="pcb_firebase_test_")
        self.storage_dir = os.path.join(self.test_dir, "storage")
        self.db_dir = os.path.join(self.test_dir, "db")
        self.auth_db_path = os.path.join(self.test_dir, "users.json")

        self.config = FirebaseConfig()
        # Force mock fallback for isolated unit testing
        self.config.mock_fallback = True

        self.firebase = FirebaseService(config=self.config, force_reinit=True)
        self.auth_service = FirebaseAuthService(
            firebase_service=self.firebase, local_auth_db_path=self.auth_db_path
        )
        self.storage_service = FirebaseStorageService(
            firebase_service=self.firebase, local_storage_root=self.storage_dir
        )
        self.analysis_service = AnalysisService(
            firebase_service=self.firebase, local_db_root=self.db_dir
        )

    def tearDown(self):
        """Clean up isolated test directories."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    # -------------------------------------------------------------------------
    # 1. Firebase Initialization & Config Tests
    # -------------------------------------------------------------------------
    def test_01_firebase_config_redaction(self):
        """Verify Firebase configuration hides sensitive details in dict export."""
        d = self.config.to_dict()
        self.assertIn("project_id", d)
        self.assertIn("storage_bucket", d)
        self.assertIn("web_api_key_configured", d)
        self.assertNotIn("GEMINI_API_KEY", d)

    def test_02_firebase_health_check(self):
        """Verify health check dictionary returns expected diagnostic fields."""
        status = self.firebase.health_check()
        self.assertIn("is_connected", status)
        self.assertIn("firestore_ready", status)
        self.assertIn("storage_ready", status)
        self.assertIn("mock_fallback_active", status)

    # -------------------------------------------------------------------------
    # 2. Authentication Tests
    # -------------------------------------------------------------------------
    def test_03_auth_registration_and_login_flow(self):
        """Test user registration and subsequent login."""
        email = "engineer@example.com"
        password = "SecurePassword123!"

        # 1. Register user
        reg_result = self.auth_service.register_user(email=email, password=password, display_name="Lead Engineer")
        self.assertTrue(reg_result["success"], f"Registration failed: {reg_result.get('error')}")
        self.assertIsNotNone(reg_result["user"])
        uid = reg_result["user"]["uid"]
        self.assertEqual(reg_result["user"]["email"], email)

        # 2. Duplicate registration should fail
        dup_result = self.auth_service.register_user(email=email, password=password)
        self.assertFalse(dup_result["success"])
        self.assertIn("EXISTS", dup_result.get("error", ""))

        # 3. Login with correct password
        login_result = self.auth_service.login_user(email=email, password=password)
        self.assertTrue(login_result["success"])
        self.assertEqual(login_result["user"]["uid"], uid)
        self.assertIsNotNone(login_result["id_token"])

        # 4. Login with incorrect password
        bad_login = self.auth_service.login_user(email=email, password="WrongPassword")
        self.assertFalse(bad_login["success"])

        # 5. Token verification
        token_info = self.auth_service.verify_token(login_result["id_token"])
        self.assertIsNotNone(token_info)
        self.assertEqual(token_info["uid"], uid)

    def test_04_auth_validation_errors(self):
        """Verify input validation for invalid email and short password."""
        res_invalid_email = self.auth_service.register_user("not-an-email", "123456")
        self.assertFalse(res_invalid_email["success"])

        res_short_pwd = self.auth_service.register_user("valid@email.com", "123")
        self.assertFalse(res_short_pwd["success"])

    # -------------------------------------------------------------------------
    # 3. Storage Hierarchy Tests
    # -------------------------------------------------------------------------
    def test_05_storage_logical_hierarchy_paths(self):
        """Verify structured directory format for input, reference, and processed images."""
        user_id = "user_42"
        analysis_id = "ANA_9901"

        # 1. Synthetic test image array
        synthetic_img = np.zeros((100, 100, 3), dtype=np.uint8)

        # 2. Upload Input
        input_upload = self.storage_service.upload_input_image(
            user_id=user_id, analysis_id=analysis_id, file_source=synthetic_img, filename="pcb_test.png"
        )
        self.assertEqual(
            input_upload["storage_path"],
            f"pcb_uploads/{user_id}/{analysis_id}/input/pcb_test.png",
        )
        self.assertTrue(os.path.exists(input_upload["local_cache_path"]))

        # 3. Upload Reference
        ref_upload = self.storage_service.upload_reference_image(
            user_id=user_id, analysis_id=analysis_id, file_source=synthetic_img, filename="pcb_golden.png"
        )
        self.assertEqual(
            ref_upload["storage_path"],
            f"pcb_uploads/{user_id}/{analysis_id}/reference/pcb_golden.png",
        )

        # 4. Upload Processed
        proc_upload = self.storage_service.upload_processed_image(
            user_id=user_id, analysis_id=analysis_id, file_source=synthetic_img, filename="annotated.png"
        )
        self.assertEqual(
            proc_upload["storage_path"],
            f"pcb_uploads/{user_id}/{analysis_id}/processed/annotated.png",
        )

        # 5. Delete analysis files
        del_success = self.storage_service.delete_analysis_files(user_id=user_id, analysis_id=analysis_id)
        self.assertTrue(del_success)
        self.assertFalse(os.path.exists(input_upload["local_cache_path"]))

    # -------------------------------------------------------------------------
    # 4. Firestore Analysis Documents & Faults Tests
    # -------------------------------------------------------------------------
    def test_06_fault_record_normalization(self):
        """Verify FaultRecord validation and schema normalization."""
        fault = FaultRecord(
            fault_id="FLT-001",
            fault_type="missing_component",
            location={"bbox": [100, 200, 150, 250], "ref": "C14"},
            confidence=1.45,  # Should clamp to 1.0
            severity="critical_open",
            evidence="Silkscreen shows C14 pad with no component body.",
            possible_cause="Feeder pick error.",
            impact="Voltage rail ripple instability.",
            recommended_action=["Install 10uF 0805 capacitor.", "Inspect solder joints."],
        )
        d = fault.to_dict()
        self.assertEqual(d["fault_id"], "FLT-001")
        self.assertEqual(d["fault_type"], "missing_component")
        self.assertEqual(d["confidence"], 1.0)
        self.assertEqual(len(d["evidence"]), 1)
        self.assertEqual(len(d["recommended_action"]), 2)

    def test_07_analysis_lifecycle_crud(self):
        """Test full Analysis lifecycle: create -> update status -> save results -> query -> delete."""
        user_id = "user_88"
        analysis_id = "ANA_TEST_007"

        # 1. Create analysis document
        doc = self.analysis_service.create_analysis(
            user_id=user_id,
            input_image_url="http://storage.example.com/input.png",
            reference_image_url="http://storage.example.com/ref.png",
            analysis_id=analysis_id,
        )
        self.assertEqual(doc["analysis_id"], analysis_id)
        self.assertEqual(doc["status"], STATUS_PENDING)
        self.assertEqual(doc["faults"], [])

        # 2. Update status to processing
        self.analysis_service.update_analysis_status(analysis_id, STATUS_PROCESSING)
        updated = self.analysis_service.get_analysis(analysis_id)
        self.assertEqual(updated["status"], STATUS_PROCESSING)

        # 3. Save detection and diagnostic results
        faults = [
            {
                "fault_id": "FLT-01",
                "fault_type": "missing_component",
                "location": {"bbox": [10, 20, 30, 40]},
                "confidence": 0.95,
                "severity": "high",
                "possible_cause": "Feeder misfeed",
                "impact": "Open circuit",
                "recommended_action": "Solder replacement capacitor",
            },
            {
                "fault_id": "FLT-02",
                "fault_type": "broken_copper_track",
                "location": {"bbox": [70, 80, 90, 100]},
                "confidence": 0.91,
                "severity": "critical_open",
                "possible_cause": "Mechanical scratch",
                "impact": "Power loss to MCU",
                "recommended_action": "Jumper wire repair per IPC-7721",
            },
        ]
        diag_report = {
            "summary": "Dual defect detected: missing C14 and fractured 3.3V power bus.",
            "root_cause": "Mechanical stress and assembly misalignment.",
        }
        recommendations = ["Solder 10uF cap", "Jumper 30 AWG wire"]

        final_doc = self.analysis_service.save_analysis_results(
            analysis_id=analysis_id,
            faults=faults,
            overall_result=RESULT_DEFECTIVE,
            confidence=0.93,
            processing_time=1.42,
            diagnostic_report=diag_report,
            recommendations=recommendations,
        )

        self.assertEqual(final_doc["status"], STATUS_COMPLETED)
        self.assertEqual(final_doc["overall_result"], RESULT_DEFECTIVE)
        self.assertEqual(len(final_doc["faults"]), 2)
        self.assertEqual(final_doc["faults"][0]["fault_type"], "missing_component")
        self.assertEqual(final_doc["faults"][1]["fault_type"], "broken_copper_track")

        # 4. List user analyses
        user_list = self.analysis_service.list_user_analyses(user_id=user_id)
        self.assertEqual(len(user_list), 1)
        self.assertEqual(user_list[0]["analysis_id"], analysis_id)

        # 5. Compute user stats
        stats = self.analysis_service.get_user_statistics(user_id=user_id)
        self.assertEqual(stats["total_scans"], 1)
        self.assertEqual(stats["defective_scans"], 1)
        self.assertEqual(stats["total_defects_found"], 2)
        self.assertEqual(stats["defect_type_breakdown"]["missing_component"], 1)
        self.assertEqual(stats["defect_type_breakdown"]["broken_copper_track"], 1)

        # 6. Delete analysis
        deleted = self.analysis_service.delete_analysis(analysis_id)
        self.assertTrue(deleted)
        self.assertIsNone(self.analysis_service.get_analysis(analysis_id))


if __name__ == "__main__":
    unittest.main()
