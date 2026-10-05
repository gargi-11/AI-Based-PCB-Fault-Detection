"""
Services Package for AI-Based PCB Fault Detection.

Provides clean modular services for:
- Firebase initialization & environment resolution (firebase_service.py)
- User Authentication (auth_service.py)
- Cloud & Local Image Storage (storage_service.py)
- Firestore Analysis Documents & Fault Records (analysis_service.py)
"""

from src.services.firebase_service import (
    FirebaseConfig,
    FirebaseService,
    get_firebase_service,
)
from src.services.auth_service import (
    FirebaseAuthService,
    UserProfile,
)
from src.services.storage_service import (
    FirebaseStorageService,
)
from src.services.analysis_service import (
    AnalysisService,
    FaultRecord,
    STATUS_PENDING,
    STATUS_PROCESSING,
    STATUS_COMPLETED,
    STATUS_FAILED,
    RESULT_PASS,
    RESULT_DEFECTIVE,
    RESULT_CRITICAL,
    RESULT_PENDING,
)

__all__ = [
    "FirebaseConfig",
    "FirebaseService",
    "get_firebase_service",
    "FirebaseAuthService",
    "UserProfile",
    "FirebaseStorageService",
    "AnalysisService",
    "FaultRecord",
    "STATUS_PENDING",
    "STATUS_PROCESSING",
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "RESULT_PASS",
    "RESULT_DEFECTIVE",
    "RESULT_CRITICAL",
    "RESULT_PENDING",
]
