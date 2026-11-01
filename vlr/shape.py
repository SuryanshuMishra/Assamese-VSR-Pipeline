"""
Module 5: Shape Features
==========================
Mouth shape measurements computed from normalised mouth landmarks.

Per-frame features:
    - Mouth width, height, opening
    - Aspect ratio
    - Inner mouth area (Shoelace formula)
    - Perimeter, roundness, symmetry

Temporal derivatives:
    - Δ(feature), Δ²(feature) for each measurement
"""

import numpy as np
import pandas as pd


def compute_shape_features(norm_mouth_sequence, timestamps):
    """
    Compute mouth shape measurements across all frames.

    Parameters:
        norm_mouth_sequence: list of (20, 2) arrays (normalised mouth landmarks)
        timestamps: list of floats, actual timestamp in seconds for each frame

    Returns:
        DataFrame with shape features + temporal derivatives
    """
    n_frames = len(norm_mouth_sequence)

    dt = np.zeros(n_frames)
    if n_frames > 1:
        dt[1:] = np.diff(timestamps)
    dt[dt == 0] = 1e-6

    records = []
    for t in range(n_frames):
        lm = norm_mouth_sequence[t]  # (20, 2)
        # Outer lip: indices 0-11 (landmarks 48-59)
        # Inner lip: indices 12-19 (landmarks 60-67)
        row = {"frame": t}

        # ---- Width: outer corners (48, 54) → indices 0, 6 ----
        width = np.linalg.norm(lm[6] - lm[0])
        row["mouth_width"] = width

        # ---- Height: top/bottom midpoints (51, 57) → indices 3, 9 ----
        height = np.linalg.norm(lm[9] - lm[3])
        row["mouth_height"] = height

        # ---- Opening: mean distance between inner lip pairs ----
        # Inner lip pairs: (61,67)→(13,19), (62,66)→(14,18), (63,65)→(15,17)
        opening_pairs = [
            np.linalg.norm(lm[13] - lm[19]),
            np.linalg.norm(lm[14] - lm[18]),
            np.linalg.norm(lm[15] - lm[17]),
        ]
        row["mouth_opening"] = np.mean(opening_pairs)

        # ---- Aspect ratio ----
        row["aspect_ratio"] = width / height if height > 0 else 0.0

        # ---- Inner mouth area (Shoelace formula) ----
        inner = lm[12:20]
        area = _polygon_area(inner)
        row["inner_area"] = area

        # ---- Perimeter of inner lip ----
        perimeter = _polygon_perimeter(inner)
        row["inner_perimeter"] = perimeter

        # ---- Roundness: 4π × area / perimeter² ----
        row["roundness"] = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0

        # ---- Symmetry: compare left and right halves of outer lip ----
        row["symmetry"] = _compute_symmetry(lm)

        records.append(row)

    df = pd.DataFrame(records)

    # ---- Temporal derivatives ----
    feature_cols = [c for c in df.columns if c != "frame"]
    for col in feature_cols:
        vals = df[col].values
        # First derivative (rate of change)
        d1 = np.zeros_like(vals)
        d1[1:] = (vals[1:] - vals[:-1]) / dt[1:]
        df[f"d_{col}"] = d1

        # Second derivative
        d2 = np.zeros_like(vals)
        d2[1:] = (d1[1:] - d1[:-1]) / dt[1:]
        df[f"dd_{col}"] = d2

    return df


def _polygon_area(points):
    """Compute area of a polygon using the Shoelace formula."""
    if len(points) < 3:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _polygon_perimeter(points):
    """Compute perimeter of a polygon."""
    diffs = np.diff(np.vstack([points, points[0:1]]), axis=0)
    return np.sqrt((diffs**2).sum(axis=1)).sum()


def _compute_symmetry(mouth_lm):
    """
    Measure left-right symmetry of outer lip. Lower = more symmetric.
    Compares mirrored left half with right half.
    """
    # Outer lip: 12 points (indices 0-11, landmarks 48-59)
    # Point 0 = left corner (48), Point 6 = right corner (54)
    outer = mouth_lm[:12]

    left_corner  = outer[0]
    right_corner = outer[6]
    midpoint = (left_corner + right_corner) / 2.0

    # Left half: points 1,2,3,4,5 (between corners, going up)
    # Right half mirror: points 11,10,9,8,7 (between corners, going down)
    left_half  = outer[1:6]
    right_half = outer[11:6:-1]

    right_mirrored = right_half.copy()
    right_mirrored[:, 0] = 2 * midpoint[0] - right_mirrored[:, 0]

    dists = np.sqrt(((left_half - right_mirrored)**2).sum(axis=1))
    return dists.mean()
