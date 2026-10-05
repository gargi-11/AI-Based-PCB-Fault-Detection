"""
Cloud Firestore Analysis Service for PCB Fault Detection.

Manages persistence, querying, and schema validation for:
- Root Collection: 'analyses'
- Document Schema:
  - analysis_id: str
  - user_id: str
  - created_at: str (ISO-8601 UTC)
  - input_image_url: str
  - reference_image_url: Optional[str]
  - status: str ('pending' | 'processing' | 'completed' | 'failed')
  - faults: List[Dict]
  - overall_result: str ('PASS' | 'DEFECTIVE' | 'CRITICAL_FAULT' | 'PENDING')
  - confidence: float
  - processing_time: float (seconds)
  - diagnostic_report: Dict[str, Any]
  - recommendations: List[str]

- Fault Item Schema:
  - fault_id: str
  - fault_type: str
  - location: Any (bbox or coordinates dict)
  - confidence: float
  - severity: str
  - evidence: List[str] | str
  - possible_cause: str
  - impact: str
  - recommended_action: List[str] | str
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from src.services.firebase_service import FirebaseService, get_firebase_service

logger = logging.getLogger(__name__)

# Valid analysis status values
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

# Valid overall results
RESULT_PASS = "PASS"
RESULT_DEFECTIVE = "DEFECTIVE"
RESULT_CRITICAL = "CRITICAL_FAULT"
RESULT_PENDING = "PENDING"


class FaultRecord:
    """Represents a validated individual PCB defect record."""

    def __init__(
        self,
        fault_id: Optional[str] = None,
        fault_type: str = "unknown_defect",
        location: Optional[Any] = None,
        confidence: float = 0.0,
        severity: str = "moderate",
        evidence: Optional[Any] = None,
        possible_cause: str = "",
        impact: str = "",
        recommended_action: Optional[Any] = None,
    ):
        self.fault_id = fault_id or f"FLT-{uuid.uuid4().hex[:6].upper()}"
        self.fault_type = str(fault_type).strip()
        self.location = location if location is not None else {}
        self.confidence = float(max(0.0, min(1.0, confidence)))
        self.severity = str(severity).strip()

        # Normalize evidence to list of strings
        if isinstance(evidence, list):
            self.evidence = [str(e) for e in evidence]
        elif isinstance(evidence, str) and evidence:
            self.evidence = [evidence]
        else:
            self.evidence = []

        self.possible_cause = str(possible_cause).strip()
        self.impact = str(impact).strip()

        # Normalize recommended action
        if isinstance(recommended_action, list):
            self.recommended_action = [str(a) for a in recommended_action]
        elif isinstance(recommended_action, str) and recommended_action:
            self.recommended_action = [recommended_action]
        else:
            self.recommended_action = []

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to Firestore-compatible dictionary."""
        return {
            "fault_id": self.fault_id,
            "fault_type": self.fault_type,
            "location": self.location,
            "confidence": round(self.confidence, 4),
            "severity": self.severity,
            "evidence": self.evidence,
            "possible_cause": self.possible_cause,
            "impact": self.impact,
            "recommended_action": self.recommended_action,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FaultRecord":
        return cls(
            fault_id=data.get("fault_id"),
            fault_type=data.get("fault_type", "unknown_defect"),
            location=data.get("location"),
            confidence=data.get("confidence", 0.0),
            severity=data.get("severity", "moderate"),
            evidence=data.get("evidence"),
            possible_cause=data.get("possible_cause", ""),
            impact=data.get("impact", ""),
            recommended_action=data.get("recommended_action"),
        )


class AnalysisService:
    """
    Manages PCB Inspection Analysis documents in Cloud Firestore with automatic
    offline local file persistence fallback.
    """

    COLLECTION_NAME = "analyses"

    def __init__(
        self,
        firebase_service: Optional[FirebaseService] = None,
        local_db_root: Optional[str] = None,
    ):
        self.firebase = firebase_service or get_firebase_service()
        self.local_root = Path(
            local_db_root
            or (Path(__file__).resolve().parent.parent.parent / "results" / "analyses")
        )
        self.local_root.mkdir(parents=True, exist_ok=True)

    def _get_local_file(self, analysis_id: str) -> Path:
        return self.local_root / f"{analysis_id}.json"

    def _save_local(self, doc_data: Dict[str, Any]) -> None:
        """Save analysis document to local disk cache."""
        analysis_id = doc_data["analysis_id"]
        file_path = self._get_local_file(analysis_id)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(doc_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to write local analysis file {file_path}: {e}")

    def create_analysis(
        self,
        user_id: str,
        input_image_url: str,
        reference_image_url: Optional[str] = None,
        analysis_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create and persist a new PCB analysis record with 'pending' status.

        Args:
            user_id: ID of the user submitting the inspection.
            input_image_url: Cloud or local URL of the input PCB image.
            reference_image_url: Optional golden reference image URL.
            analysis_id: Optional pre-generated ID (auto-generated if None).
            metadata: Optional additional metadata dict.

        Returns:
            Dict containing the initialized analysis document.
        """
        aid = analysis_id or f"ANA-{uuid.uuid4().hex[:8].upper()}"
        now_utc = datetime.now(timezone.utc).isoformat()

        doc_data: Dict[str, Any] = {
            "analysis_id": aid,
            "user_id": str(user_id).strip(),
            "created_at": now_utc,
            "input_image_url": str(input_image_url),
            "reference_image_url": str(reference_image_url) if reference_image_url else None,
            "status": STATUS_PENDING,
            "faults": [],
            "overall_result": RESULT_PENDING,
            "confidence": 0.0,
            "processing_time": 0.0,
            "diagnostic_report": {},
            "recommendations": [],
            "metadata": metadata or {},
        }

        # 1. Save to Firestore if connected
        db = self.firebase.get_firestore()
        if db is not None:
            try:
                db.collection(self.COLLECTION_NAME).document(aid).set(doc_data)
                logger.info(f"Created analysis {aid} in Cloud Firestore.")
            except Exception as e:
                logger.warning(f"Firestore create failed for {aid}: {e}. Saving locally.")

        # 2. Always persist locally
        self._save_local(doc_data)
        return doc_data

    def update_analysis_status(
        self, analysis_id: str, status: str, error_message: Optional[str] = None
    ) -> bool:
        """Update the processing status of an analysis."""
        current = self.get_analysis(analysis_id)
        if not current:
            return False

        current["status"] = status
        current["updated_at"] = datetime.now(timezone.utc).isoformat()
        if error_message:
            current["error_message"] = error_message

        db = self.firebase.get_firestore()
        if db is not None:
            try:
                db.collection(self.COLLECTION_NAME).document(analysis_id).update(
                    {"status": status, "updated_at": current["updated_at"], "error_message": error_message}
                )
            except Exception as e:
                logger.warning(f"Firestore update status failed for {analysis_id}: {e}")

        self._save_local(current)
        return True

    def save_analysis_results(
        self,
        analysis_id: str,
        faults: List[Dict[str, Any]],
        overall_result: str = RESULT_PASS,
        confidence: float = 1.0,
        processing_time: float = 0.0,
        diagnostic_report: Optional[Dict[str, Any]] = None,
        recommendations: Optional[List[str]] = None,
        processed_image_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Record final detection and diagnostic results for an analysis session.
        """
        current = self.get_analysis(analysis_id) or {
            "analysis_id": analysis_id,
            "user_id": "unknown",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_image_url": "",
        }

        # Validate fault records
        normalized_faults = []
        for item in faults:
            if isinstance(item, FaultRecord):
                normalized_faults.append(item.to_dict())
            elif isinstance(item, dict):
                normalized_faults.append(FaultRecord.from_dict(item).to_dict())

        # Determine overall result if not explicitly defective/pass
        if not overall_result or overall_result == RESULT_PENDING:
            overall_result = RESULT_DEFECTIVE if normalized_faults else RESULT_PASS

        current["status"] = STATUS_COMPLETED
        current["completed_at"] = datetime.now(timezone.utc).isoformat()
        current["faults"] = normalized_faults
        current["overall_result"] = overall_result
        current["confidence"] = round(float(confidence), 4)
        current["processing_time"] = round(float(processing_time), 3)
        current["diagnostic_report"] = diagnostic_report or {}
        current["recommendations"] = recommendations or []

        if processed_image_url:
            current["processed_image_url"] = processed_image_url
        if metadata:
            current.setdefault("metadata", {}).update(metadata)

        # Save to Cloud Firestore
        db = self.firebase.get_firestore()
        if db is not None:
            try:
                db.collection(self.COLLECTION_NAME).document(analysis_id).set(current)
                logger.info(f"Saved analysis results for {analysis_id} in Cloud Firestore.")
            except Exception as e:
                logger.warning(f"Firestore save results failed for {analysis_id}: {e}")

        # Save locally
        self._save_local(current)
        return current

    def get_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an analysis document by its unique ID."""
        # 1. Check Firestore
        db = self.firebase.get_firestore()
        if db is not None:
            try:
                doc = db.collection(self.COLLECTION_NAME).document(analysis_id).get()
                if doc.exists:
                    data = doc.to_dict()
                    self._save_local(data)
                    return data
            except Exception as e:
                logger.warning(f"Firestore fetch failed for {analysis_id}: {e}")

        # 2. Fallback to local cache
        file_path = self._get_local_file(analysis_id)
        if file_path.is_file():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to read local file {file_path}: {e}")

        return None

    def list_user_analyses(
        self, user_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        List all analyses belonging to a specific user, sorted from newest to oldest.
        """
        results: List[Dict[str, Any]] = []

        # 1. Fetch from Firestore
        db = self.firebase.get_firestore()
        if db is not None:
            try:
                query = (
                    db.collection(self.COLLECTION_NAME)
                    .where("user_id", "==", str(user_id).strip())
                    .limit(limit)
                )
                docs = query.stream()
                for doc in docs:
                    d = doc.to_dict()
                    results.append(d)
                # Sort descending by created_at
                results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
                return results
            except Exception as e:
                logger.warning(f"Firestore list query failed for user {user_id}: {e}. Querying local cache.")

        # 2. Fallback to local disk scan
        for f in self.local_root.glob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    if data.get("user_id") == str(user_id).strip():
                        results.append(data)
            except Exception:
                continue

        results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return results[:limit]

    def delete_analysis(self, analysis_id: str) -> bool:
        """Delete an analysis document from Firestore and local cache."""
        success = True
        db = self.firebase.get_firestore()
        if db is not None:
            try:
                db.collection(self.COLLECTION_NAME).document(analysis_id).delete()
            except Exception as e:
                logger.error(f"Firestore delete failed for {analysis_id}: {e}")
                success = False

        file_path = self._get_local_file(analysis_id)
        if file_path.is_file():
            try:
                file_path.unlink()
            except Exception as e:
                logger.error(f"Local delete failed for {file_path}: {e}")
                success = False

        return success

    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Compute inspection summary statistics for a user dashboard."""
        analyses = self.list_user_analyses(user_id=user_id, limit=500)
        total = len(analyses)
        if total == 0:
            return {
                "total_scans": 0,
                "passed_scans": 0,
                "defective_scans": 0,
                "critical_faults": 0,
                "pass_rate_pct": 100.0,
                "total_defects_found": 0,
                "defect_type_breakdown": {},
            }

        passed = sum(1 for a in analyses if a.get("overall_result") == RESULT_PASS)
        defective = sum(1 for a in analyses if a.get("overall_result") == RESULT_DEFECTIVE)
        critical = sum(1 for a in analyses if a.get("overall_result") == RESULT_CRITICAL)

        total_defects = 0
        breakdown: Dict[str, int] = {}
        for a in analyses:
            for fault in a.get("faults", []):
                total_defects += 1
                ftype = fault.get("fault_type", "other")
                breakdown[ftype] = breakdown.get(ftype, 0) + 1

        pass_rate = (passed / total * 100.0) if total > 0 else 100.0

        return {
            "total_scans": total,
            "passed_scans": passed,
            "defective_scans": defective,
            "critical_faults": critical,
            "pass_rate_pct": round(pass_rate, 2),
            "total_defects_found": total_defects,
            "defect_type_breakdown": breakdown,
        }
