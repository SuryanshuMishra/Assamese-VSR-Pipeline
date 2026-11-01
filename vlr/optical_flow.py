"""
Module 6: Image-Level Motion
==============================
Frame differencing and Farneback dense optical flow on normalised
mouth ROI images.

Features per frame:
    - Frame difference (MAD, max diff, std)
    - Optical flow: mean/max horizontal, vertical, magnitude
    - Motion energy: sum of squared flow magnitudes
    - Horizontal/vertical energy breakdown
"""

import numpy as np
import cv2
import pandas as pd


def compute_optical_flow_features(mouth_images, timestamps, params=None):
    """
    Compute dense optical flow features (Farneback) on normalised mouth crops.

    Parameters:
        mouth_images: list of grayscale mouth crop images
        timestamps: list of floats, actual timestamp in seconds for each frame
        params: dict of parameters for cv2.calcOpticalFlowFarneback

    Returns:
        DataFrame with optical flow features per frame
    """
    if params is None:
        params = dict(pyr_scale=0.5, levels=3, winsize=15,
                      iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    n_frames = len(mouth_images)
    records = []
    prev_gray = None

    for t in range(n_frames):
        curr_gray = mouth_images[t]
        row = {"frame": t}

        if prev_gray is None:
            row.update(_zero_flow_record())
            records.append(row)
            prev_gray = curr_gray
            continue

        # Frame differencing
        diff = cv2.absdiff(curr_gray, prev_gray).astype(np.float64)
        row["frame_diff_mean"] = diff.mean()
        row["frame_diff_max"]  = float(diff.max())
        row["frame_diff_std"]  = diff.std()

        # Dense optical flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, flow=None, **params
        )
        flow_x   = flow[:, :, 0]
        flow_y   = flow[:, :, 1]
        flow_mag = np.sqrt(flow_x**2 + flow_y**2)

        row["flow_x_mean"]   = flow_x.mean()
        row["flow_y_mean"]   = flow_y.mean()
        row["flow_mag_mean"] = flow_mag.mean()
        row["flow_x_max"]    = float(np.abs(flow_x).max())
        row["flow_y_max"]    = float(np.abs(flow_y).max())
        row["flow_mag_max"]  = float(flow_mag.max())

        row["motion_energy"]     = float((flow_mag**2).sum())
        row["horizontal_energy"] = float((flow_x**2).sum())
        row["vertical_energy"]   = float((flow_y**2).sum())

        records.append(row)
        prev_gray = curr_gray

    return pd.DataFrame(records)


def compute_flow_visualization(prev_gray, curr_gray):
    """
    Compute dense optical flow and return an HSV colour visualisation.

    Parameters:
        prev_gray, curr_gray: consecutive grayscale mouth images

    Returns:
        flow_bgr: BGR visualisation of optical flow
        flow: raw (H, W, 2) flow array
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, flow=None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )

    mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
    hsv = np.zeros((*prev_gray.shape, 3), dtype=np.uint8)
    hsv[:, :, 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), flow


def _zero_flow_record():
    """Zero-value dict for frame 0 (no previous frame)."""
    return {
        "frame_diff_mean": 0.0, "frame_diff_max": 0.0, "frame_diff_std": 0.0,
        "flow_x_mean": 0.0, "flow_y_mean": 0.0, "flow_mag_mean": 0.0,
        "flow_x_max": 0.0, "flow_y_max": 0.0, "flow_mag_max": 0.0,
        "motion_energy": 0.0, "horizontal_energy": 0.0, "vertical_energy": 0.0,
    }
