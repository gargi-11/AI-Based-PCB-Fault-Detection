"""
Firebase Authentication Service for PCB Fault Detection.

Supports:
- User Registration (Email & Password)
- User Login & Session Verification
- ID Token Verification via Firebase Admin SDK
- Password Reset Requests
- Safe Local / Offline Fallback for Development & Testing
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
import uuid
import requests

from src.services.firebase_service import FirebaseService, get_firebase_service

# Optional firebase_admin auth import
try:
    from firebase_admin import auth as admin_auth
    ADMIN_AUTH_AVAILABLE = True
except ImportError:
    admin_auth = None
    ADMIN_AUTH_AVAILABLE = False

logger = logging.getLogger(__name__)

FIREBASE_AUTH_SIGNIN_URL = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
FIREBASE_AUTH_SIGNUP_URL = "https://identitytoolkit.googleapis.com/v1/accounts:signUp"
FIREBASE_AUTH_RESET_URL = "https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode"


class UserProfile:
    """Represents an authenticated user profile."""

    def __init__(
        self,
        uid: str,
        email: str,
        display_name: Optional[str] = None,
        id_token: Optional[str] = None,
        is_mock: bool = False,
    ):
        self.uid = uid
        self.email = email
        self.display_name = display_name or email.split("@")[0]
        self.id_token = id_token
        self.is_mock = is_mock

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "email": self.email,
            "display_name": self.display_name,
            "is_mock": self.is_mock,
        }


class FirebaseAuthService:
    """
    Manages Firebase User Authentication with client REST API for login/registration
    and Admin SDK for server-side token validation, plus local fallback authentication.
    """

    def __init__(
        self,
        firebase_service: Optional[FirebaseService] = None,
        local_auth_db_path: Optional[str] = None,
    ):
        self.firebase = firebase_service or get_firebase_service()
        self.local_db_path = Path(
            local_auth_db_path
            or (Path(__file__).resolve().parent.parent.parent / "results" / "auth" / "users.json")
        )
        self.local_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_local_db()

    def _init_local_db(self) -> None:
        if not self.local_db_path.is_file():
            try:
                with open(self.local_db_path, "w", encoding="utf-8") as f:
                    json.dump({}, f)
            except Exception as e:
                logger.error(f"Failed to initialize local auth db: {e}")

    def _hash_password(self, password: str) -> str:
        """Hash password for local mock store."""
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    def register_user(
        self,
        email: str,
        password: str,
        display_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Register a new user with email and password.

        Returns:
            Dict with 'success', 'user' (UserProfile dict), and optional 'error'.
        """
        email_clean = email.strip().lower()
        if not email_clean or "@" not in email_clean:
            return {"success": False, "user": None, "error": "Invalid email address."}
        if len(password) < 6:
            return {"success": False, "user": None, "error": "Password must be at least 6 characters."}

        api_key = self.firebase.config.web_api_key

        # 1. Cloud registration via Firebase Auth REST API if Web API key configured
        if api_key:
            try:
                payload = {
                    "email": email_clean,
                    "password": password,
                    "returnSecureToken": True,
                }
                resp = requests.post(f"{FIREBASE_AUTH_SIGNUP_URL}?key={api_key}", json=payload, timeout=10)
                data = resp.json()
                if resp.status_code == 200:
                    uid = data["localId"]
                    id_token = data.get("idToken")
                    user = UserProfile(
                        uid=uid,
                        email=email_clean,
                        display_name=display_name or data.get("displayName"),
                        id_token=id_token,
                        is_mock=False,
                    )
                    return {"success": True, "user": user.to_dict(), "id_token": id_token, "error": None}
                else:
                    error_msg = data.get("error", {}).get("message", "Registration failed.")
                    return {"success": False, "user": None, "error": error_msg}
            except Exception as e:
                logger.warning(f"Firebase REST signup failed: {e}. Trying Admin/Local.")

        # 2. Cloud registration via Firebase Admin SDK
        if ADMIN_AUTH_AVAILABLE and self.firebase.is_connected:
            try:
                user_record = admin_auth.create_user(
                    email=email_clean,
                    password=password,
                    display_name=display_name,
                )
                user = UserProfile(
                    uid=user_record.uid,
                    email=user_record.email,
                    display_name=user_record.display_name,
                    is_mock=False,
                )
                return {"success": True, "user": user.to_dict(), "error": None}
            except Exception as e:
                logger.warning(f"Firebase Admin create_user failed: {e}")
                return {"success": False, "user": None, "error": str(e)}

        # 3. Local fallback registration
        try:
            with open(self.local_db_path, "r", encoding="utf-8") as f:
                users = json.load(f)
        except Exception:
            users = {}

        if email_clean in users:
            return {"success": False, "user": None, "error": "EMAIL_EXISTS: User already registered."}

        mock_uid = f"usr_{uuid.uuid4().hex[:12]}"
        users[email_clean] = {
            "uid": mock_uid,
            "email": email_clean,
            "password_hash": self._hash_password(password),
            "display_name": display_name or email_clean.split("@")[0],
        }

        try:
            with open(self.local_db_path, "w", encoding="utf-8") as f:
                json.dump(users, f, indent=2)
        except Exception as e:
            return {"success": False, "user": None, "error": f"Failed to save user: {e}"}

        user = UserProfile(
            uid=mock_uid,
            email=email_clean,
            display_name=display_name,
            id_token=f"mock_token_{mock_uid}",
            is_mock=True,
        )
        return {"success": True, "user": user.to_dict(), "id_token": user.id_token, "error": None}

    def login_user(self, email: str, password: str) -> Dict[str, Any]:
        """
        Authenticate user with email and password.

        Returns:
            Dict with 'success', 'user' (UserProfile dict), 'id_token', and optional 'error'.
        """
        email_clean = email.strip().lower()
        if not email_clean:
            return {"success": False, "user": None, "id_token": None, "error": "Email is required."}

        api_key = self.firebase.config.web_api_key

        # 1. Cloud login via Firebase Auth REST API
        if api_key:
            try:
                payload = {
                    "email": email_clean,
                    "password": password,
                    "returnSecureToken": True,
                }
                resp = requests.post(f"{FIREBASE_AUTH_SIGNIN_URL}?key={api_key}", json=payload, timeout=10)
                data = resp.json()
                if resp.status_code == 200:
                    uid = data["localId"]
                    id_token = data.get("idToken")
                    user = UserProfile(
                        uid=uid,
                        email=email_clean,
                        display_name=data.get("displayName") or email_clean.split("@")[0],
                        id_token=id_token,
                        is_mock=False,
                    )
                    return {"success": True, "user": user.to_dict(), "id_token": id_token, "error": None}
                else:
                    error_msg = data.get("error", {}).get("message", "Invalid credentials.")
                    return {"success": False, "user": None, "id_token": None, "error": error_msg}
            except Exception as e:
                logger.warning(f"Firebase REST signin failed: {e}. Falling back to local.")

        # 2. Local fallback login
        try:
            with open(self.local_db_path, "r", encoding="utf-8") as f:
                users = json.load(f)
        except Exception:
            users = {}

        if email_clean not in users:
            return {"success": False, "user": None, "id_token": None, "error": "EMAIL_NOT_FOUND"}

        stored = users[email_clean]
        if stored.get("password_hash") != self._hash_password(password):
            return {"success": False, "user": None, "id_token": None, "error": "INVALID_PASSWORD"}

        mock_token = f"mock_token_{stored['uid']}"
        user = UserProfile(
            uid=stored["uid"],
            email=email_clean,
            display_name=stored.get("display_name"),
            id_token=mock_token,
            is_mock=True,
        )
        return {"success": True, "user": user.to_dict(), "id_token": mock_token, "error": None}

    def verify_token(self, id_token: str) -> Optional[Dict[str, Any]]:
        """
        Verify a Firebase ID token using the Admin SDK or mock validator.
        """
        if not id_token:
            return None

        # Check mock token
        if id_token.startswith("mock_token_"):
            mock_uid = id_token.replace("mock_token_", "")
            return {"uid": mock_uid, "email": f"{mock_uid}@example.com", "is_mock": True}

        if ADMIN_AUTH_AVAILABLE and self.firebase.is_connected:
            try:
                decoded = admin_auth.verify_id_token(id_token)
                return decoded
            except Exception as e:
                logger.warning(f"Firebase token verification failed: {e}")
                return None

        return None

    def send_password_reset(self, email: str) -> Dict[str, Any]:
        """Send password reset email."""
        email_clean = email.strip().lower()
        api_key = self.firebase.config.web_api_key

        if api_key:
            try:
                payload = {"requestType": "PASSWORD_RESET", "email": email_clean}
                resp = requests.post(f"{FIREBASE_AUTH_RESET_URL}?key={api_key}", json=payload, timeout=10)
                if resp.status_code == 200:
                    return {"success": True, "error": None}
                else:
                    return {"success": False, "error": resp.json().get("error", {}).get("message")}
            except Exception as e:
                return {"success": False, "error": str(e)}

        return {"success": True, "message": "Password reset simulation recorded."}
