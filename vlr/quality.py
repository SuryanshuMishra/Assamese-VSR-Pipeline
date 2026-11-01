"""
Module 7: Quality Metrics & Validity
======================================
Per-frame quality assessment: blur, brightness, face confidence,
ROI statistics, and an explicit validity mask with failure reasons.

Aggregate summary: detection rate, validity rate, longest failure run,
crop-centre jitter, scale jitter.
"""

import numpy as np
import cv2
import pandas as pd


def compute_quality_metrics(frame_bgr, landmark_result, norm_result,
                            blur_threshold=50.0, brightness_range=(30, 230)):
    """
    Compute quality metrics for a single frame.

    Parameters:
        frame_bgr: original BGR frame
        landmark_result: LandmarkResult from detection module
        norm_result: NormResult from normalisation module (or None)
        blur_threshold: Laplacian variance below this = blurry
        brightness_range: (min, max) acceptable mean brightness

    Returns:
        dict with quality metrics and validity flag
    """
    row = {
        "frame": landmark_result.frame_index,
        "timestamp_sec": landmark_result.timestamp_sec,
        "detected": landmark_result.detected,
        "face_confidence": landmark_result.confidence,
        "failure_reason": landmark_result.failure_reason or "",
    }

    if not landmark_result.detected or norm_result is None:
        row.update({
            "blur_score": 0.0, "brightness": 0.0,
            "roi_width": 0, "roi_height": 0, "roi_area": 0,
            "crop_center_x": 0.0, "crop_center_y": 0.0,
            "face_angle_deg": 0.0, "valid": False,
        })
        if not row["failure_reason"]:
            row["failure_reason"] = "normalization_failed"
        return row

    # Blur: Laplacian variance on mouth image
    mouth_img = norm_result.mouth_image
    laplacian  = cv2.Laplacian(mouth_img, cv2.CV_64F)
    blur_score = laplacian.var()
    row["blur_score"] = round(blur_score, 2)

    # Brightness: mean pixel value of mouth image
    brightness = float(mouth_img.mean())
    row["brightness"] = round(brightness, 2)

    # ROI statistics
    if landmark_result.face_rect:
        l, t, r, b = landmark_result.face_rect
        row["roi_width"]  = r - l
        row["roi_height"] = b - t
        row["roi_area"]   = (r - l) * (b - t)
    else:
        row["roi_width"] = row["roi_height"] = row["roi_area"] = 0

    # Normalisation metadata
    row["crop_center_x"] = round(norm_result.crop_center[0], 2)
    row["crop_center_y"] = round(norm_result.crop_center[1], 2)
    row["face_angle_deg"]= round(norm_result.face_angle_deg, 2)

    # Validity check
    reasons = []
    if blur_score < blur_threshold:
        reasons.append("too_blurry")
    if brightness < brightness_range[0]:
        reasons.append("too_dark")
    if brightness > brightness_range[1]:
        reasons.append("too_bright")

    row["valid"] = len(reasons) == 0
    if reasons:
        row["failure_reason"] = ";".join(reasons)

    return row


def build_quality_log(quality_records):
    """
    Build quality log DataFrame and compute summary statistics.

    Parameters:
        quality_records: list of dicts from compute_quality_metrics

    Returns:
        df: DataFrame — full quality log
        summary: dict — aggregate quality statistics
    """
    df = pd.DataFrame(quality_records)

    total    = len(df)
    detected = int(df["detected"].sum())
    valid    = int(df["valid"].sum()) if "valid" in df.columns else 0

    # Longest failure run
    longest_fail = _longest_run(~df["detected"].values)

    # Crop-centre jitter
    if "crop_center_x" in df.columns and detected > 1:
        mask = df["detected"].values
        cx = df["crop_center_x"].values.copy(); cx[~mask] = np.nan
        cy = df["crop_center_y"].values.copy(); cy[~mask] = np.nan
        dcx = np.diff(cx[~np.isnan(cx)])
        dcy = np.diff(cy[~np.isnan(cy)])
        jitter = np.sqrt(dcx**2 + dcy**2) if len(dcx) > 0 else np.array([0])
        crop_jitter_mean   = float(np.nanmean(jitter))
        crop_jitter_median = float(np.nanmedian(jitter))
    else:
        crop_jitter_mean = crop_jitter_median = 0.0

    summary = {
        "total_frames": total,
        "detected_frames": detected,
        "valid_frames": valid,
        "detection_rate": f"{(detected / total * 100):.1f}%" if total > 0 else "0%",
        "validity_rate": f"{(valid / total * 100):.1f}%" if total > 0 else "0%",
        "failed_frames": total - detected,
        "longest_failure_run": longest_fail,
        "crop_jitter_mean": round(crop_jitter_mean, 4),
        "crop_jitter_median": round(crop_jitter_median, 4),
    }

    return df, summary


def _longest_run(bool_array):
    """Compute the longest consecutive True run in a boolean array."""
    max_run = current = 0
    for v in bool_array:
        if v:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run
