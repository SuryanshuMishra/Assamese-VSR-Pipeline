"""
Module 8: Visualisation
========================
Preview video generation and static analysis plots.

Preview video layout (4 panels):
    ┌────────────────┬──────────────┬──────────────────┬──────────────────┐
    │ Original frame │ Normalised   │ Displacement     │ Optical flow     │
    │ + 68 landmarks │ mouth ROI    │ vectors overlay  │ visualisation    │
    └────────────────┴──────────────┴──────────────────┴──────────────────┘

Static plots:
    - Mouth opening curve
    - Motion energy over time
    - Lip motion magnitude
    - Keypoint trajectories
    - Displacement heatmap
    - Raw vs. normalised comparison
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


# ==========================================
# Preview Video
# ==========================================

def draw_landmarks_on_frame(frame, landmarks_68, mouth_indices=range(48, 68)):
    """Draw all 68 landmarks, highlighting mouth points."""
    vis = frame.copy()
    for i, (x, y) in enumerate(landmarks_68):
        color  = (0, 255, 0) if i in mouth_indices else (180, 180, 180)
        radius = 2 if i in mouth_indices else 1
        cv2.circle(vis, (int(x), int(y)), radius, color, -1)

    outer = landmarks_68[48:60].astype(np.int32)
    cv2.polylines(vis, [outer], isClosed=True, color=(0, 255, 0), thickness=1)
    inner = landmarks_68[60:68].astype(np.int32)
    cv2.polylines(vis, [inner], isClosed=True, color=(0, 200, 255), thickness=1)
    return vis


def draw_displacement_vectors(mouth_image, norm_mouth_prev, norm_mouth_curr,
                              scale=5.0):
    """
    Draw per-keypoint displacement arrows on a mouth ROI image.

    Parameters:
        mouth_image: grayscale mouth crop
        norm_mouth_prev: (20, 2) previous frame normalised mouth (or None)
        norm_mouth_curr: (20, 2) current frame normalised mouth
        scale: arrow length multiplier
    """
    h, w = mouth_image.shape[:2]
    vis = cv2.cvtColor(mouth_image, cv2.COLOR_GRAY2BGR)

    if norm_mouth_prev is None:
        return vis

    x_min, y_min = norm_mouth_curr.min(axis=0)
    x_max, y_max = norm_mouth_curr.max(axis=0)
    pad = 0.3
    rx, ry = x_max - x_min, y_max - y_min
    x_min -= rx * pad; x_max += rx * pad
    y_min -= ry * pad; y_max += ry * pad

    def to_px(pt):
        px = int((pt[0] - x_min) / (x_max - x_min) * w)
        py = int((pt[1] - y_min) / (y_max - y_min) * h)
        return (px, py)

    for i in range(len(norm_mouth_curr)):
        p = to_px(norm_mouth_curr[i])
        d = norm_mouth_curr[i] - norm_mouth_prev[i]
        end = (int(p[0] + d[0] * scale), int(p[1] + d[1] * scale))
        cv2.circle(vis, p, 2, (0, 255, 0), -1)
        cv2.arrowedLine(vis, p, end, (0, 0, 255), 1, tipLength=0.3)

    return vis


def create_preview_frame(original_bgr, landmarks_68, mouth_image,
                         disp_vis, flow_vis, frame_idx, timestamp,
                         panel_h=240):
    """Compose a 4-panel preview frame."""
    aspect = original_bgr.shape[1] / original_bgr.shape[0]
    panel_w_main = int(panel_h * aspect)
    panel_w_sq   = panel_h

    p1 = draw_landmarks_on_frame(original_bgr, landmarks_68)
    p1 = cv2.resize(p1, (panel_w_main, panel_h))

    p2 = cv2.cvtColor(mouth_image, cv2.COLOR_GRAY2BGR)
    p2 = cv2.resize(p2, (panel_w_sq, panel_h), interpolation=cv2.INTER_NEAREST)

    p3 = cv2.resize(disp_vis, (panel_w_sq, panel_h),
                    interpolation=cv2.INTER_NEAREST)

    if flow_vis is not None:
        p4 = cv2.resize(flow_vis, (panel_w_sq, panel_h),
                        interpolation=cv2.INTER_NEAREST)
    else:
        p4 = np.zeros((panel_h, panel_w_sq, 3), dtype=np.uint8)

    combined = np.hstack([p1, p2, p3, p4])

    bar = np.zeros((30, combined.shape[1], 3), dtype=np.uint8)
    text = f"Frame {frame_idx} | t={timestamp:.3f}s"
    labels = ("Original + Landmarks    |    Normalised Mouth"
              "    |    Displacement    |    Optical Flow")
    cv2.putText(bar, text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(bar, labels, (panel_w_main - 50, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    return np.vstack([combined, bar])


# ==========================================
# Static Plots
# ==========================================

def plot_mouth_opening(shape_df, output_path):
    """Plot mouth opening over time."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(shape_df["frame"], shape_df["mouth_opening"],
            color="#2196F3", linewidth=0.8)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mouth Opening (normalised units)")
    ax.set_title("Mouth Opening Over Time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def plot_motion_energy(flow_df, output_path):
    """Plot motion energy over time."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(flow_df["frame"], flow_df["motion_energy"],
                    alpha=0.4, color="#FF5722")
    ax.plot(flow_df["frame"], flow_df["motion_energy"],
            color="#FF5722", linewidth=0.8)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Motion Energy")
    ax.set_title("Optical Flow Motion Energy Over Time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def plot_lip_motion_magnitude(motion_df, output_path):
    """Plot mean displacement (lip motion magnitude) over time."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(motion_df["frame"], motion_df["mean_disp"],
                    alpha=0.3, color="#9C27B0")
    ax.plot(motion_df["frame"], motion_df["mean_disp"],
            color="#9C27B0", linewidth=1.2)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean Displacement (normalised)")
    ax.set_title("Lip Motion Magnitude Over Time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def plot_trajectories(norm_mouth_sequence, output_path,
                      keypoints=(0, 3, 6, 9, 14, 18)):
    """Plot X and Y trajectories for selected keypoints."""
    positions = np.array(norm_mouth_sequence)
    labels = {0: "Left corner (48)", 3: "Top mid (51)", 6: "Right corner (54)",
              9: "Bottom mid (57)", 14: "Inner top (62)", 18: "Inner bottom (66)"}

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ki in keypoints:
        label = labels.get(ki, f"Point {ki}")
        axes[0].plot(positions[:, ki, 0], linewidth=0.8, label=label)
        axes[1].plot(positions[:, ki, 1], linewidth=0.8, label=label)

    axes[0].set_ylabel("X (normalised)")
    axes[0].set_title("Landmark X Trajectories")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Y (normalised)")
    axes[1].set_xlabel("Frame")
    axes[1].set_title("Landmark Y Trajectories")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def plot_displacement_heatmap(motion_df_raw, output_path, n_points=20):
    """Plot displacement magnitude as a heatmap (keypoints × time)."""
    n_frames = len(motion_df_raw)
    heatmap = np.zeros((n_points, n_frames))
    for i in range(n_points):
        col = f"disp_mag_{i}"
        if col in motion_df_raw.columns:
            heatmap[i] = motion_df_raw[col].values

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(heatmap, aspect="auto", cmap="hot", interpolation="nearest")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mouth Keypoint")
    ax.set_title("Displacement Magnitude Heatmap")
    ax.set_yticks(range(0, n_points, 2))
    fig.colorbar(im, ax=ax, label="Displacement magnitude")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def plot_raw_vs_normalized(raw_mouth_seq, norm_mouth_seq, output_path,
                           keypoint=3):
    """Compare raw vs normalised trajectories for a single keypoint."""
    raw  = np.array(raw_mouth_seq)
    norm = np.array(norm_mouth_seq)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].plot(raw[:, keypoint, 0], linewidth=0.8, color="#F44336")
    axes[0, 0].set_title("Raw X (pixel coords)")
    axes[0, 0].set_ylabel("X")
    axes[0, 0].grid(True, alpha=0.3)

    axes[1, 0].plot(raw[:, keypoint, 1], linewidth=0.8, color="#F44336")
    axes[1, 0].set_title("Raw Y (pixel coords)")
    axes[1, 0].set_ylabel("Y")
    axes[1, 0].set_xlabel("Frame")
    axes[1, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(norm[:, keypoint, 0], linewidth=0.8, color="#4CAF50")
    axes[0, 1].set_title("Normalised X")
    axes[0, 1].set_ylabel("X")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 1].plot(norm[:, keypoint, 1], linewidth=0.8, color="#4CAF50")
    axes[1, 1].set_title("Normalised Y")
    axes[1, 1].set_ylabel("Y")
    axes[1, 1].set_xlabel("Frame")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"Raw vs Normalised — Keypoint {keypoint} (top midpoint)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
