"""
PCB Reference Alignment & Homography Registration Module.

Aligns a test PCB photograph with a golden/reference template PCB image
using ORB/SIFT keypoint extraction and RANSAC perspective transformation.
"""

import logging
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def align_images_orb(
    test_image_bgr: np.ndarray,
    ref_image_bgr: np.ndarray,
    max_features: int = 2000,
    match_ratio: float = 0.75,
    min_inliers: int = 10,
) -> Dict[str, Any]:
    """
    Geometrically align test_image_bgr to ref_image_bgr coordinate space.

    Args:
        test_image_bgr: Test PCB image (BGR ndarray).
        ref_image_bgr: Reference/Golden PCB image (BGR ndarray).
        max_features: Maximum ORB keypoints to detect.
        match_ratio: Lowe's ratio test threshold for feature matching.
        min_inliers: Minimum number of RANSAC inliers required for valid registration.

    Returns:
        Dict containing:
        - "success": bool
        - "aligned_image": np.ndarray (test image warped into ref coordinates)
        - "homography_matrix": Optional[np.ndarray] (3x3 matrix)
        - "inliers_count": int
        - "match_count": int
        - "alignment_confidence": float (0.0 to 1.0)
        - "method": str
        - "error": Optional[str]
    """
    if test_image_bgr is None or ref_image_bgr is None:
        return {
            "success": False,
            "aligned_image": test_image_bgr,
            "homography_matrix": None,
            "inliers_count": 0,
            "match_count": 0,
            "alignment_confidence": 0.0,
            "method": "none",
            "error": "One or both input images are None.",
        }

    # Target output dimensions from reference image
    ref_h, ref_w = ref_image_bgr.shape[:2]
    test_h, test_w = test_image_bgr.shape[:2]

    # Convert to grayscale for feature detection
    test_gray = cv2.cvtColor(test_image_bgr, cv2.COLOR_BGR2GRAY) if test_image_bgr.ndim == 3 else test_image_bgr
    ref_gray = cv2.cvtColor(ref_image_bgr, cv2.COLOR_BGR2GRAY) if ref_image_bgr.ndim == 3 else ref_image_bgr

    # 1. Initialize ORB detector
    orb = cv2.ORB_create(
        nfeatures=max_features,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
    )

    kp_test, des_test = orb.detectAndCompute(test_gray, None)
    kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)

    if des_test is None or des_ref is None or len(kp_test) < 4 or len(kp_ref) < 4:
        logger.warning("Insufficient keypoints detected for ORB alignment.")
        # Fallback to direct resize if identical aspect ratio
        resized_fallback = cv2.resize(test_image_bgr, (ref_w, ref_h), interpolation=cv2.INTER_AREA)
        return {
            "success": False,
            "aligned_image": resized_fallback,
            "homography_matrix": None,
            "inliers_count": 0,
            "match_count": 0,
            "alignment_confidence": 0.0,
            "method": "resize_fallback",
            "error": "Insufficient keypoints detected for feature matching.",
        }

    # 2. Match descriptors with BFMatcher using Hamming distance
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        raw_matches = matcher.knnMatch(des_test, des_ref, k=2)
    except Exception as e:
        logger.warning(f"Feature matching error: {e}")
        raw_matches = []

    # 3. Apply Lowe's ratio test
    good_matches = []
    for m_pair in raw_matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < match_ratio * n.distance:
                good_matches.append(m)

    if len(good_matches) < min_inliers:
        logger.warning(f"Found only {len(good_matches)} matches (minimum required: {min_inliers}).")
        resized_fallback = cv2.resize(test_image_bgr, (ref_w, ref_h), interpolation=cv2.INTER_AREA)
        return {
            "success": False,
            "aligned_image": resized_fallback,
            "homography_matrix": None,
            "inliers_count": 0,
            "match_count": len(good_matches),
            "alignment_confidence": 0.0,
            "method": "resize_fallback",
            "error": f"Insufficient good matches: {len(good_matches)} < {min_inliers}",
        }

    # 4. Extract location of good matches
    src_pts = np.float32([kp_test[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 5. Compute Homography matrix with RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        logger.warning("Homography computation failed.")
        resized_fallback = cv2.resize(test_image_bgr, (ref_w, ref_h), interpolation=cv2.INTER_AREA)
        return {
            "success": False,
            "aligned_image": resized_fallback,
            "homography_matrix": None,
            "inliers_count": 0,
            "match_count": len(good_matches),
            "alignment_confidence": 0.0,
            "method": "resize_fallback",
            "error": "Homography estimation failed.",
        }

    inliers_count = int(np.sum(mask)) if mask is not None else 0
    inlier_ratio = (inliers_count / len(good_matches)) if good_matches else 0.0
    confidence = float(max(0.0, min(1.0, inlier_ratio * min(1.0, inliers_count / 30.0))))

    # 6. Warp test image into reference perspective
    aligned_image = cv2.warpPerspective(
        test_image_bgr,
        H,
        (ref_w, ref_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    return {
        "success": bool(inliers_count >= min_inliers),
        "aligned_image": aligned_image,
        "homography_matrix": H,
        "inliers_count": inliers_count,
        "match_count": len(good_matches),
        "alignment_confidence": round(confidence, 4),
        "method": "ORB_RANSAC_Homography",
        "error": None if inliers_count >= min_inliers else "Low inlier count",
    }
