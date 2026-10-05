"""
Firebase Cloud Storage Service for PCB Fault Detection.

Manages logical storage hierarchy:
- pcb_uploads/{user_id}/{analysis_id}/input/{filename}
- pcb_uploads/{user_id}/{analysis_id}/reference/{filename}
- pcb_uploads/{user_id}/{analysis_id}/processed/{filename}

Provides upload, download, URL generation, and automatic local caching/fallback.
"""

import io
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from src.services.firebase_service import FirebaseService, get_firebase_service

logger = logging.getLogger(__name__)


class FirebaseStorageService:
    """
    Manages image asset persistence on Firebase Cloud Storage with structured directories:
    pcb_uploads/{user_id}/{analysis_id}/{category}/{filename}
    """

    def __init__(
        self,
        firebase_service: Optional[FirebaseService] = None,
        local_storage_root: Optional[str] = None,
    ):
        self.firebase = firebase_service or get_firebase_service()
        self.local_root = Path(
            local_storage_root
            or (Path(__file__).resolve().parent.parent.parent / "results" / "storage")
        )
        self.local_root.mkdir(parents=True, exist_ok=True)

    def _build_storage_path(
        self, user_id: str, analysis_id: str, category: str, filename: str
    ) -> str:
        """Construct standard logical cloud storage path."""
        # Sanitize parts
        clean_user = user_id.strip() or "anonymous_user"
        clean_analysis = analysis_id.strip() or "default_analysis"
        clean_cat = category.strip().lower()
        clean_name = Path(filename).name or "image.png"

        return f"pcb_uploads/{clean_user}/{clean_analysis}/{clean_cat}/{clean_name}"

    def _convert_to_bytes(
        self, file_source: Union[str, Path, bytes, np.ndarray, Image.Image], default_format: str = "PNG"
    ) -> Tuple[bytes, str]:
        """Convert various input types into bytes and determine MIME content type."""
        if isinstance(file_source, bytes):
            return file_source, "image/png"

        if isinstance(file_source, (str, Path)):
            path = Path(file_source)
            if not path.is_file():
                raise FileNotFoundError(f"Source image file not found: {path}")
            data = path.read_bytes()
            suffix = path.suffix.lower()
            mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
            return data, mime

        if isinstance(file_source, np.ndarray):
            # Convert OpenCV BGR to RGB if 3 channels
            if file_source.ndim == 3 and file_source.shape[2] == 3:
                rgb = cv2.cvtColor(file_source, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
            elif file_source.ndim == 2:
                pil_img = Image.fromarray(file_source)
            else:
                pil_img = Image.fromarray(file_source)
            buf = io.BytesIO()
            pil_img.save(buf, format=default_format)
            return buf.getvalue(), f"image/{default_format.lower()}"

        if isinstance(file_source, Image.Image):
            buf = io.BytesIO()
            file_source.save(buf, format=default_format)
            return buf.getvalue(), f"image/{default_format.lower()}"

        raise ValueError(f"Unsupported image source type: {type(file_source)}")

    def upload_file(
        self,
        storage_path: str,
        file_source: Union[str, Path, bytes, np.ndarray, Image.Image],
        content_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upload an asset to Firebase Cloud Storage. If offline, stores in local storage.

        Args:
            storage_path: Relative path in storage (e.g. pcb_uploads/u1/a1/input/pcb.png).
            file_source: File path, bytes, numpy array, or PIL Image.
            content_type: Optional MIME type override.

        Returns:
            Dict containing:
            - "storage_path": str
            - "url": str (cloud public/signed URL or local file URI)
            - "is_cloud": bool
            - "size_bytes": int
        """
        data_bytes, detected_mime = self._convert_to_bytes(file_source)
        mime = content_type or detected_mime
        size = len(data_bytes)

        # 1. Always save a local copy in local_root cache
        local_dest = self.local_root / storage_path
        local_dest.parent.mkdir(parents=True, exist_ok=True)
        local_dest.write_bytes(data_bytes)
        local_url = str(local_dest.resolve())

        # 2. Upload to Firebase Storage bucket if available
        bucket = self.firebase.get_storage_bucket()
        if bucket is not None:
            try:
                blob = bucket.blob(storage_path)
                blob.upload_from_string(data_bytes, content_type=mime)
                try:
                    blob.make_public()
                    cloud_url = blob.public_url
                except Exception:
                    # If bucket doesn't allow public ACLs, use signed URL or media URL
                    cloud_url = blob.generate_signed_url(version="v4", expiration=3600, method="GET")

                logger.info(f"Uploaded {storage_path} to Firebase Cloud Storage.")
                return {
                    "storage_path": storage_path,
                    "url": cloud_url,
                    "is_cloud": True,
                    "size_bytes": size,
                    "local_cache_path": local_url,
                }
            except Exception as e:
                logger.warning(f"Firebase Storage upload failed for {storage_path}: {e}. Falling back to local.")

        return {
            "storage_path": storage_path,
            "url": local_url,
            "is_cloud": False,
            "size_bytes": size,
            "local_cache_path": local_url,
        }

    def upload_input_image(
        self,
        user_id: str,
        analysis_id: str,
        file_source: Union[str, Path, bytes, np.ndarray, Image.Image],
        filename: str = "input_board.png",
    ) -> Dict[str, Any]:
        """Upload raw input test PCB image."""
        storage_path = self._build_storage_path(user_id, analysis_id, "input", filename)
        return self.upload_file(storage_path, file_source)

    def upload_reference_image(
        self,
        user_id: str,
        analysis_id: str,
        file_source: Union[str, Path, bytes, np.ndarray, Image.Image],
        filename: str = "golden_reference.png",
    ) -> Dict[str, Any]:
        """Upload golden/reference PCB image."""
        storage_path = self._build_storage_path(user_id, analysis_id, "reference", filename)
        return self.upload_file(storage_path, file_source)

    def upload_processed_image(
        self,
        user_id: str,
        analysis_id: str,
        file_source: Union[str, Path, bytes, np.ndarray, Image.Image],
        filename: str = "annotated_defects.png",
    ) -> Dict[str, Any]:
        """Upload preprocessed or defect-annotated PCB image."""
        storage_path = self._build_storage_path(user_id, analysis_id, "processed", filename)
        return self.upload_file(storage_path, file_source)

    def get_signed_url(self, storage_path: str, expiration_minutes: int = 60) -> str:
        """Generate a time-limited signed URL for secure download."""
        bucket = self.firebase.get_storage_bucket()
        if bucket is not None:
            try:
                blob = bucket.blob(storage_path)
                return blob.generate_signed_url(
                    version="v4",
                    expiration=expiration_minutes * 60,
                    method="GET",
                )
            except Exception as e:
                logger.warning(f"Failed to generate signed URL for {storage_path}: {e}")

        # Fallback to local file path
        local_path = self.local_root / storage_path
        return str(local_path.resolve())

    def delete_analysis_files(self, user_id: str, analysis_id: str) -> bool:
        """Delete all storage files belonging to a specific analysis session."""
        prefix = f"pcb_uploads/{user_id}/{analysis_id}/"
        bucket = self.firebase.get_storage_bucket()
        success = True

        if bucket is not None:
            try:
                blobs = bucket.list_blobs(prefix=prefix)
                for blob in blobs:
                    blob.delete()
            except Exception as e:
                logger.error(f"Failed to delete cloud blobs for {prefix}: {e}")
                success = False

        # Remove local directory
        local_dir = self.local_root / "pcb_uploads" / user_id / analysis_id
        if local_dir.is_dir():
            try:
                shutil.rmtree(str(local_dir))
            except Exception as e:
                logger.error(f"Failed to delete local cache directory {local_dir}: {e}")
                success = False

        return success
