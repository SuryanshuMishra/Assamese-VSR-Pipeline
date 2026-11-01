"""
Module 2: Landmark Detection
==============================
dlib-based face detection and 68-point facial landmark extraction.
Tracks detection confidence, failure reasons, and stable reference points.

Uses pretrained shape_predictor_68_face_landmarks.dat.
"""

import bz2
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import dlib
import numpy as np


# Model download URL
SHAPE_PREDICTOR_URL      = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
SHAPE_PREDICTOR_FILENAME = "shape_predictor_68_face_landmarks.dat"

# Landmark index ranges
MOUTH_OUTER = list(range(48, 60))    # 12 outer lip points
MOUTH_INNER = list(range(60, 68))    # 8 inner lip points
MOUTH_ALL   = list(range(48, 68))    # 20 total mouth points
NOSE_BRIDGE       = [27, 28, 29, 30]
LEFT_EYE_CORNERS  = [36, 39]
RIGHT_EYE_CORNERS = [42, 45]
STABLE_REFS       = [27, 28, 29, 30, 36, 39, 42, 45]  # 8 points


@dataclass
class LandmarkResult:
    """Detection result for a single frame."""
    frame_index: int
    timestamp_sec: float
    detected: bool
    face_rect: Optional[tuple] = None           # (left, top, right, bottom)
    landmarks_68: Optional[np.ndarray] = None   # (68, 2) raw pixel coordinates
    mouth_landmarks: Optional[np.ndarray] = None  # (20, 2)
    stable_refs: Optional[np.ndarray] = None    # (8, 2) nose + eye corners
    confidence: float = 0.0                     # face area / frame area
    failure_reason: Optional[str] = None


def get_model_path():
    """Return the path to the shape predictor model, download if missing."""
    search_paths = [
        Path.cwd() / SHAPE_PREDICTOR_FILENAME,
        Path(__file__).parent.parent / SHAPE_PREDICTOR_FILENAME,
        Path.home() / ".dlib" / SHAPE_PREDICTOR_FILENAME,
    ]
    for path in search_paths:
        if path.exists():
            return str(path)

    # Download
    download_dir = Path(__file__).parent.parent
    model_path = download_dir / SHAPE_PREDICTOR_FILENAME
    compressed_path = download_dir / f"{SHAPE_PREDICTOR_FILENAME}.bz2"

    print("[INFO] Downloading shape predictor model (~100 MB)...")
    try:
        urllib.request.urlretrieve(SHAPE_PREDICTOR_URL, str(compressed_path))
        with bz2.open(str(compressed_path), "rb") as f_in:
            with open(str(model_path), "wb") as f_out:
                f_out.write(f_in.read())
        compressed_path.unlink()
        print(f"[INFO] Model saved to {model_path}")
        return str(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to download model: {e}")


class LandmarkDetector:
    """
    Wraps dlib face detector + 68-point shape predictor.
    Tracks detection confidence and failure reasons.
    """

    def __init__(self, model_path=None, min_face_fraction=0.005):
        if model_path is None:
            model_path = get_model_path()
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)
        self.min_face_fraction = min_face_fraction

    def detect(self, frame_data):
        """
        Detect face and extract landmarks from a FrameData object.

        Returns:
            LandmarkResult with confidence, failure_reason, stable_refs
        """
        bgr = frame_data.bgr
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        frame_area = h * w

        faces = self.detector(gray, 0)

        if len(faces) == 0:
            return LandmarkResult(
                frame_index=frame_data.index,
                timestamp_sec=frame_data.timestamp_sec,
                detected=False,
                failure_reason="no_face",
            )

        # Pick largest face
        face = max(faces, key=lambda r: r.width() * r.height())
        face_area = face.width() * face.height()
        confidence = face_area / frame_area

        if confidence < self.min_face_fraction:
            return LandmarkResult(
                frame_index=frame_data.index,
                timestamp_sec=frame_data.timestamp_sec,
                detected=False,
                face_rect=(face.left(), face.top(), face.right(), face.bottom()),
                confidence=confidence,
                failure_reason="face_too_small",
            )

        # Extract 68 landmarks
        shape = self.predictor(gray, face)
        landmarks_68 = np.array(
            [(shape.part(i).x, shape.part(i).y) for i in range(68)],
            dtype=np.float64,
        )

        mouth_lm = landmarks_68[MOUTH_ALL].copy()
        stable = landmarks_68[STABLE_REFS].copy()

        return LandmarkResult(
            frame_index=frame_data.index,
            timestamp_sec=frame_data.timestamp_sec,
            detected=True,
            face_rect=(face.left(), face.top(), face.right(), face.bottom()),
            landmarks_68=landmarks_68,
            mouth_landmarks=mouth_lm,
            stable_refs=stable,
            confidence=round(confidence, 6),
        )
