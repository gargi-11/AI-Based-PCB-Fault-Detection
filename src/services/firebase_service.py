"""
Central Firebase Initialization & Configuration Service.

Manages connection to Firebase Admin SDK, Cloud Firestore, and Firebase Storage.
Provides safe credentials resolution, environment validation, and graceful fallback
when running in local development mode without cloud credentials.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from dotenv import load_dotenv

# Optional firebase_admin import
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, storage
    FIREBASE_ADMIN_AVAILABLE = True
except ImportError:
    firebase_admin = None
    credentials = None
    firestore = None
    storage = None
    FIREBASE_ADMIN_AVAILABLE = False

logger = logging.getLogger(__name__)


class FirebaseConfig:
    """Encapsulates Firebase project configuration loaded from environment."""

    def __init__(self, env_path: Optional[str] = None):
        self._load_env(env_path)
        self.project_id = os.getenv("FIREBASE_PROJECT_ID", "").strip()
        self.storage_bucket = os.getenv("FIREBASE_STORAGE_BUCKET", "").strip()
        self.web_api_key = os.getenv("FIREBASE_WEB_API_KEY", "").strip()
        self.service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "").strip()
        self.mock_fallback = os.getenv("FIREBASE_MOCK_FALLBACK", "true").lower() in ("true", "1", "yes")

    def _load_env(self, env_path: Optional[str] = None) -> None:
        if env_path and Path(env_path).is_file():
            load_dotenv(dotenv_path=env_path)
        else:
            default_env = Path(__file__).resolve().parent.parent.parent / ".env"
            if default_env.is_file():
                load_dotenv(dotenv_path=default_env)
            else:
                load_dotenv()

    def to_dict(self) -> Dict[str, Any]:
        """Return safe configuration dict with secrets redacted."""
        return {
            "project_id": self.project_id or "(not set)",
            "storage_bucket": self.storage_bucket or "(not set)",
            "web_api_key_configured": bool(self.web_api_key),
            "service_account_path": self.service_account_path or "(not set)",
            "mock_fallback": self.mock_fallback,
            "firebase_admin_available": FIREBASE_ADMIN_AVAILABLE,
        }


class FirebaseService:
    """
    Singleton service managing Firebase application lifecycle,
    Firestore database client, and Cloud Storage bucket.
    """

    _instance: Optional["FirebaseService"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(FirebaseService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[FirebaseConfig] = None, force_reinit: bool = False):
        if getattr(self, "_initialized", False) and not force_reinit:
            return

        self.config = config or FirebaseConfig()
        self.app: Optional[Any] = None
        self.db: Optional[Any] = None
        self.bucket: Optional[Any] = None
        self.is_connected = False
        self.initialization_error: Optional[str] = None

        self._initialize_firebase()
        self._initialized = True

    def _initialize_firebase(self) -> None:
        """Attempt to initialize the Firebase Admin SDK."""
        if not FIREBASE_ADMIN_AVAILABLE:
            self.initialization_error = "firebase-admin library is not installed."
            logger.warning(self.initialization_error)
            return

        try:
            # Check if default app is already initialized
            existing_apps = getattr(firebase_admin, "_apps", {})
            if existing_apps and firebase_admin.DEFAULT_APP_NAME in existing_apps:
                self.app = firebase_admin.get_app()
            else:
                cred = self._resolve_credentials()
                options = {}
                if self.config.storage_bucket:
                    options["storageBucket"] = self.config.storage_bucket
                if self.config.project_id:
                    options["projectId"] = self.config.project_id

                if cred:
                    self.app = firebase_admin.initialize_app(cred, options=options)
                elif self.config.project_id:
                    # Attempt ADC / default credentials
                    self.app = firebase_admin.initialize_app(options=options)
                else:
                    self.initialization_error = "No Firebase Project ID or Service Account provided."
                    logger.info("Firebase running in offline/local fallback mode.")
                    return

            # Initialize Firestore
            try:
                self.db = firestore.client(app=self.app)
            except Exception as e:
                logger.warning(f"Could not initialize Firestore client: {e}")
                self.db = None

            # Initialize Cloud Storage
            try:
                bucket_name = self.config.storage_bucket or (f"{self.config.project_id}.appspot.com" if self.config.project_id else None)
                if bucket_name:
                    self.bucket = storage.bucket(bucket_name, app=self.app)
            except Exception as e:
                logger.warning(f"Could not initialize Storage bucket: {e}")
                self.bucket = None

            self.is_connected = True
            logger.info("Firebase Admin SDK successfully initialized.")

        except Exception as e:
            self.initialization_error = str(e)
            self.is_connected = False
            logger.warning(f"Firebase initialization failed: {e}. Fallback enabled: {self.config.mock_fallback}")

    def _resolve_credentials(self) -> Optional[Any]:
        """Resolve Firebase credentials from file path or environment."""
        if not credentials:
            return None

        # 1. Explicit Service Account Path
        if self.config.service_account_path:
            path = Path(self.config.service_account_path)
            if path.is_file():
                return credentials.Certificate(str(path))
            else:
                logger.warning(f"Specified service account file not found: {path}")

        # 2. Search common project locations
        project_root = Path(__file__).resolve().parent.parent.parent
        common_paths = [
            project_root / "serviceAccountKey.json",
            project_root / "firebase_credentials.json",
        ]
        for p in common_paths:
            if p.is_file():
                logger.info(f"Using service account key found at: {p.name}")
                return credentials.Certificate(str(p))

        # 3. Fallback to Application Default Credentials if GOOGLE_APPLICATION_CREDENTIALS set
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            return credentials.ApplicationDefault()

        return None

    def get_firestore(self) -> Optional[Any]:
        """Get the active Cloud Firestore client, or None if offline."""
        return self.db if self.is_connected else None

    def get_storage_bucket(self) -> Optional[Any]:
        """Get the active Cloud Storage bucket, or None if offline."""
        return self.bucket if self.is_connected else None

    def health_check(self) -> Dict[str, Any]:
        """Return comprehensive health status of Firebase services."""
        return {
            "is_connected": self.is_connected,
            "firestore_ready": self.db is not None,
            "storage_ready": self.bucket is not None,
            "mock_fallback_active": not self.is_connected and self.config.mock_fallback,
            "config": self.config.to_dict(),
            "error": self.initialization_error,
        }


# Global singleton accessor
def get_firebase_service(env_path: Optional[str] = None) -> FirebaseService:
    """Get or create the global FirebaseService instance."""
    config = FirebaseConfig(env_path=env_path) if env_path else None
    return FirebaseService(config=config)
