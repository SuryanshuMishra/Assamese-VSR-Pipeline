"""
Module 3: Normalisation
========================
Mean-centres mouth landmarks (subtracts centroid) and crops the mouth
ROI directly from the original frame.

Approach:
    1. Compute the mouth centroid from the 20 mouth keypoints.
    2. Subtract the centroid so landmarks are centred at the origin
       (removes translation / head position).
    3. Crop the mouth region from the original frame using the
       landmark bounding box + padding, then resize to crop_size × crop_size.

No affine warp or rotation correction — keeps the code simple while
still removing the main source of noise (absolute pixel position).
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class NormResult:
    """Normalisation output for a single frame."""
    norm_landmarks_68: np.ndarray       # (68, 2) mean-centred coordinates
    norm_mouth: np.ndarray              # (20, 2) mean-centred mouth landmarks
    mouth_image: np.ndarray             # Grayscale mouth ROI (crop_size × crop_size)
    face_angle_deg: float               # In-plane face rotation (metadata, not corrected)
    crop_center: tuple                  # (cx, cy) of mouth crop in original frame
    crop_size: tuple                    # (w, h) of output image


def mean_centre_landmarks(landmarks_68):
    """
    Mean-centre all 68 landmarks by subtracting the mouth centroid.

    Parameters:
        landmarks_68: (68, 2) raw pixel coordinates

    Returns:
        norm_68: (68, 2) centred coordinates
        norm_mouth: (20, 2) centred mouth landmarks
        centroid: (cx, cy) mouth centroid in original pixel space
    """
    mouth = landmarks_68[48:68]
    centroid = mouth.mean(axis=0)

    norm_68 = landmarks_68 - centroid       # shift everything so mouth is at origin
    norm_mouth = norm_68[48:68]

    return norm_68, norm_mouth, tuple(centroid)


def crop_mouth_roi(bgr_frame, landmarks_68, crop_size=96, padding_factor=0.3):
    """
    Crop the mouth region from the original frame using the landmark
    bounding box with padding, then resize to crop_size × crop_size.

    Parameters:
        bgr_frame: original BGR frame
        landmarks_68: (68, 2) raw landmarks
        crop_size: output square dimension
        padding_factor: extra padding around mouth bounding box

    Returns:
        mouth_image: grayscale (crop_size, crop_size)
        crop_center: (cx, cy) centre of the crop in original frame
    """
    h, w = bgr_frame.shape[:2]
    mouth = landmarks_68[48:68]

    x_min, y_min = mouth.min(axis=0)
    x_max, y_max = mouth.max(axis=0)
    mw, mh = x_max - x_min, y_max - y_min
    pad_x, pad_y = mw * padding_factor, mh * padding_factor

    x1 = max(0, int(x_min - pad_x))
    y1 = max(0, int(y_min - pad_y))
    x2 = min(w, int(x_max + pad_x))
    y2 = min(h, int(y_max + pad_y))

    crop_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    if x2 <= x1 or y2 <= y1:
        return np.zeros((crop_size, crop_size), dtype=np.uint8), crop_center

    roi = bgr_frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mouth_image = cv2.resize(gray, (crop_size, crop_size),
                             interpolation=cv2.INTER_AREA)
    return mouth_image, crop_center


def normalize_frame(frame_bgr, landmarks_68, stable_refs, crop_size=96):
    """
    Normalise one frame:
        1. Mean-centre landmarks (subtract mouth centroid).
        2. Crop mouth ROI from original frame and resize.

    Parameters:
        frame_bgr: BGR image
        landmarks_68: (68, 2) raw pixel coordinates
        stable_refs: (8, 2) stable reference points (used for angle metadata)
        crop_size: mouth ROI output size

    Returns:
        NormResult or None if normalisation fails
    """
    try:
        # --- Mean-centre landmarks ---
        norm_68, norm_mouth, centroid = mean_centre_landmarks(landmarks_68)

        # --- Crop mouth ROI ---
        mouth_img, crop_center = crop_mouth_roi(
            frame_bgr, landmarks_68, crop_size=crop_size
        )

        # --- Metadata (face angle, for logging only) ---
        left_eye_outer  = landmarks_68[36]
        right_eye_outer = landmarks_68[45]
        dx = right_eye_outer[0] - left_eye_outer[0]
        dy = right_eye_outer[1] - left_eye_outer[1]
        face_angle_deg = float(np.degrees(np.arctan2(dy, dx)))

        return NormResult(
            norm_landmarks_68=norm_68,
            norm_mouth=norm_mouth,
            mouth_image=mouth_img,
            face_angle_deg=face_angle_deg,
            crop_center=crop_center,
            crop_size=(crop_size, crop_size),
        )
    except Exception:
        return None
