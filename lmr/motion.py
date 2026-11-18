"""
Module 4: Motion Features
==========================
Computes frame-to-frame lip motion from normalised mouth landmarks.

Features per keypoint (20 points × 2 axes):
    - Displacement (dx, dy), magnitude, direction
    - Velocity, acceleration
    - Cumulative trajectory, total distance

Aggregate features per frame:
    - Mean/max displacement, velocity, acceleration
    - Motion centroid
"""

import numpy as np
import pandas as pd


def compute_motion_features(norm_mouth_sequence, timestamps, smoothing_window=3):
    """
    Compute per-keypoint motion features across all frames.

    Parameters:
        norm_mouth_sequence: list of (20, 2) arrays (normalised mouth landmarks)
        timestamps: list of floats, actual timestamp in seconds for each frame
        smoothing_window: window size for moving-average smoothing (odd number)

    Returns:
        df_raw: DataFrame with raw motion features
        df_smoothed: DataFrame with smoothed motion features
    """
    n_frames = len(norm_mouth_sequence)
    n_points = 20

    dt = np.zeros(n_frames)
    if n_frames > 1:
        dt[1:] = np.diff(timestamps)
    dt[dt == 0] = 1e-6

    dt2 = dt[:, None]        # (T, 1)
    dt3 = dt[:, None, None]  # (T, 1, 1)

    positions = np.array(norm_mouth_sequence)  # (T, 20, 2)

    # Displacement
    displacement = np.zeros_like(positions)
    displacement[1:] = positions[1:] - positions[:-1]

    dx  = displacement[:, :, 0]
    dy  = displacement[:, :, 1]
    mag = np.sqrt(dx**2 + dy**2)
    direction = np.arctan2(dy, dx)

    # Velocity
    velocity = displacement / dt3
    vel_mag  = mag / dt2

    # Acceleration
    acceleration = np.zeros_like(velocity)
    acceleration[1:] = (velocity[1:] - velocity[:-1]) / dt3[1:]
    acc_mag = np.sqrt(acceleration[:, :, 0]**2 + acceleration[:, :, 1]**2)

    # Cumulative trajectory
    cumulative_dist = np.cumsum(mag, axis=0)

    # Aggregate
    mean_disp = mag.mean(axis=1)
    max_disp  = mag.max(axis=1)
    mean_vel  = vel_mag.mean(axis=1)
    max_vel   = vel_mag.max(axis=1)
    mean_acc  = acc_mag.mean(axis=1)

    # Motion centroid
    motion_cx = np.zeros(n_frames)
    motion_cy = np.zeros(n_frames)
    for t in range(n_frames):
        total_mag = mag[t].sum()
        if total_mag > 0:
            motion_cx[t] = (positions[t, :, 0] * mag[t]).sum() / total_mag
            motion_cy[t] = (positions[t, :, 1] * mag[t]).sum() / total_mag

    # Build DataFrame
    records = []
    for t in range(n_frames):
        row = {"frame": t}
        for i in range(n_points):
            row[f"dx_{i}"]       = dx[t, i]
            row[f"dy_{i}"]       = dy[t, i]
            row[f"disp_mag_{i}"] = mag[t, i]
            row[f"disp_dir_{i}"] = direction[t, i]
            row[f"vel_x_{i}"]    = velocity[t, i, 0]
            row[f"vel_y_{i}"]    = velocity[t, i, 1]
            row[f"vel_mag_{i}"]  = vel_mag[t, i]
            row[f"acc_x_{i}"]    = acceleration[t, i, 0]
            row[f"acc_y_{i}"]    = acceleration[t, i, 1]
            row[f"acc_mag_{i}"]  = acc_mag[t, i]
            row[f"cum_dist_{i}"] = cumulative_dist[t, i]
        row["mean_disp"]  = mean_disp[t]
        row["max_disp"]   = max_disp[t]
        row["mean_vel"]   = mean_vel[t]
        row["max_vel"]    = max_vel[t]
        row["mean_acc"]   = mean_acc[t]
        row["motion_cx"]  = motion_cx[t]
        row["motion_cy"]  = motion_cy[t]
        records.append(row)

    df_raw = pd.DataFrame(records)

    # Smoothed version
    df_smoothed = df_raw.copy()
    numeric_cols = [c for c in df_smoothed.columns if c != "frame"]
    for col in numeric_cols:
        df_smoothed[col] = _moving_average(df_smoothed[col].values, smoothing_window)

    return df_raw, df_smoothed


def compute_total_distance(norm_mouth_sequence):
    """
    Total distance travelled by each of the 20 keypoints.

    Returns:
        total_dist: (20,) array — total path length per keypoint
    """
    positions = np.array(norm_mouth_sequence)
    disp = np.diff(positions, axis=0)
    mag  = np.sqrt((disp**2).sum(axis=2))
    return mag.sum(axis=0)


def _moving_average(signal, window):
    """Centred moving average smoothing."""
    if window < 2 or len(signal) < window:
        return signal.copy()
    kernel = np.ones(window) / window
    padded = np.pad(signal, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(signal)]
