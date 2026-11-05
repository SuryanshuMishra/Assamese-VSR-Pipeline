"""
Linguistic Analysis & Word Boundary Alignment for Assamese VSR
==============================================================
Correlates kinematic lip features with ground-truth speech boundaries
from s320_1351_moi_doi_khalu_u1.lab:
  0.000s - 4.390s: sil (pre-speech silence)
  4.390s - 5.000s: moi (মই - "I")
  5.000s - 5.650s: doi (দৈ - "curd")
  5.650s - 6.353s: khalu (খালোঁ - "ate")
  6.353s - 7.855s: sil (post-speech silence)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Paths
out_dir = Path("s320_1351_output")
csv_dir = out_dir / "csvs"
graphs_dir = out_dir / "graphs"
graphs_dir.mkdir(parents=True, exist_ok=True)

# 1. Load data
shape_df = pd.read_csv(csv_dir / "shape_features.csv")
motion_df = pd.read_csv(csv_dir / "motion_features.csv")
lm_df = pd.read_csv(csv_dir / "landmarks.csv")
flow_df = pd.read_csv(csv_dir / "optical_flow_features.csv")

fps = 60.0
time_sec = lm_df["timestamp_sec"].values
n_frames = len(time_sec)

# 2. Ground-truth word boundaries (from .lab file)
segments = [
    {"label": "Silence (sil)", "word": "sil", "assamese": "নিৰৱতা", "start": 0.000, "end": 4.390, "color": "#E0E0E0"},
    {"label": "Word 1: moi", "word": "moi", "assamese": "মই", "start": 4.390, "end": 5.000, "color": "#BBDEFB"},
    {"label": "Word 2: doi", "word": "doi", "assamese": "দৈ", "start": 5.000, "end": 5.650, "color": "#C8E6C9"},
    {"label": "Word 3: khalu", "word": "khalu", "assamese": "খালোঁ", "start": 5.650, "end": 6.353, "color": "#FFE0B2"},
    {"label": "Silence (sil)", "word": "sil", "assamese": "নিৰৱতা", "start": 6.353, "end": 7.855, "color": "#E0E0E0"},
]

speech_start = 4.390
speech_end = 6.353

# Helper to shade segments
def add_segment_shading(ax, show_labels=True):
    for seg in segments:
        ax.axvspan(seg["start"], seg["end"], color=seg["color"], alpha=0.45, zorder=0)
        ax.axvline(seg["start"], color="#555555", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(segments[-1]["end"], color="#555555", linestyle="--", linewidth=1.0, alpha=0.7)
    
    if show_labels:
        for seg in segments[1:4]: # moi, doi, khalu
            mid = (seg["start"] + seg["end"]) / 2.0
            ax.text(mid, ax.get_ylim()[1] * 0.92, f"{seg['word']}\n({seg['assamese']})", 
                    ha="center", va="top", fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=seg["color"], alpha=0.85))

# ====================================================================
# Plot 1: Mouth Aperture & Expansion vs Acoustic Boundaries
# ====================================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Subplot 1: Mouth Opening & Inner Area
ax = axes[0]
l1, = ax.plot(time_sec, shape_df["mouth_opening"], color="#D32F2F", linewidth=2.0, label="Mouth Opening (Aperture px)")
ax_twin = ax.twinx()
l2, = ax_twin.plot(time_sec, shape_df["inner_area"], color="#7B1FA2", linewidth=1.8, linestyle="-.", label="Inner Lip Area (Shoelace px²)")
ax.set_ylabel("Mouth Opening (px)", color="#D32F2F", fontweight="bold")
ax_twin.set_ylabel("Inner Area (px²)", color="#7B1FA2", fontweight="bold")
ax.set_title("1. Assamese Viseme Boundaries: Mouth Aperture & Area vs Acoustic Words", fontsize=13, fontweight="bold")
add_segment_shading(ax)
ax.grid(True, alpha=0.3)
lines = [l1, l2]
ax.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=9)

# Subplot 2: Mouth Width (Spreading) & Roundness
ax = axes[1]
l1, = ax.plot(time_sec, shape_df["mouth_width"], color="#1976D2", linewidth=2.0, label="Mouth Width (Lip Spreading px)")
ax_twin = ax.twinx()
l2, = ax_twin.plot(time_sec, shape_df["roundness"], color="#388E3C", linewidth=1.8, linestyle=":", label="Lip Roundness (Circularity)")
ax.set_ylabel("Mouth Width (px)", color="#1976D2", fontweight="bold")
ax_twin.set_ylabel("Roundness (4πA/P²)", color="#388E3C", fontweight="bold")
ax.set_title("2. Horizontal Lip Spreading vs Rounding During Articulation", fontsize=11, fontweight="bold")
add_segment_shading(ax, show_labels=False)
ax.grid(True, alpha=0.3)
lines = [l1, l2]
ax.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=9)

# Subplot 3: First Temporal Derivative (Rate of Opening / Velocity)
ax = axes[2]
ax.plot(time_sec, shape_df["d_mouth_opening"], color="#E65100", linewidth=1.5, label="Δ Mouth Opening (Velocity px/s)")
ax.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
ax.set_ylabel("Δ Opening (px/s)", color="#E65100", fontweight="bold")
ax.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
ax.set_title("3. Aperture Velocity Inflections & Boundary Trigger Points", fontsize=11, fontweight="bold")
add_segment_shading(ax, show_labels=False)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", fontsize=9)

fig.tight_layout()
fig.savefig(str(graphs_dir / "linguistic_aperture_alignment.png"), dpi=200)
plt.close(fig)
print("[OK] Saved linguistic_aperture_alignment.png")

# ====================================================================
# Plot 2: Displacement Heatmap with Acoustic Boundary Overlay
# ====================================================================
fig, ax = plt.subplots(figsize=(14, 6))

# Construct heatmap (20 keypoints x time)
n_points = 20
heatmap = np.zeros((n_points, n_frames))
for pt in range(n_points):
    col = f"disp_mag_{pt}"
    if col in motion_df.columns:
        heatmap[pt, :] = motion_df[col].values

im = ax.imshow(heatmap, aspect="auto", cmap="inferno", interpolation="nearest",
               extent=[time_sec[0], time_sec[-1], n_points - 0.5, -0.5])

# Shading and boundary lines
for seg in segments:
    ax.axvspan(seg["start"], seg["end"], color=seg["color"], alpha=0.25)
    ax.axvline(seg["start"], color="cyan", linestyle="--", linewidth=1.5, alpha=0.9)
ax.axvline(segments[-1]["end"], color="cyan", linestyle="--", linewidth=1.5, alpha=0.9)

# Labels
for seg in segments[1:4]:
    mid = (seg["start"] + seg["end"]) / 2.0
    ax.text(mid, -1.2, f"{seg['word']} ({seg['assamese']})", 
            ha="center", va="bottom", fontsize=11, fontweight="bold", color="#1A237E",
            bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="cyan", alpha=0.9))

ax.set_yticks(range(0, n_points, 2))
ax.set_yticklabels([f"KP {i}" for i in range(0, n_points, 2)])
ax.set_ylabel("Mouth Keypoint (0-11 Outer, 12-19 Inner)", fontsize=11, fontweight="bold")
ax.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
ax.set_title("Kinematic Displacement Heatmap Overlaid with Ground-Truth Speech Boundaries", fontsize=13, fontweight="bold")
fig.colorbar(im, ax=ax, label="Displacement Magnitude (px)")
ax.set_ylim(n_points - 0.5, -2.0) # space for labels

fig.tight_layout()
fig.savefig(str(graphs_dir / "linguistic_heatmap_alignment.png"), dpi=200)
plt.close(fig)
print("[OK] Saved linguistic_heatmap_alignment.png")

# ====================================================================
# Plot 3: Visual Lead Time & Kinetic Energy vs Acoustic Speech Window
# ====================================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

# Subplot 1: Lip Velocity
ax = axes[0]
ax.plot(time_sec, motion_df["mean_vel"], color="#00897B", linewidth=2.0, label="Mean Lip Velocity (px/s)")
ax.plot(time_sec, motion_df["max_vel"], color="#004D40", linewidth=1.2, linestyle="--", label="Peak Lip Velocity (px/s)")
ax.set_ylabel("Velocity (px/s)", fontweight="bold")
ax.set_title("Visual Movement Lead Time: Lip Velocity Accelerates Before Audio Onset", fontsize=12, fontweight="bold")
add_segment_shading(ax)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left")

# Detect visual onset (when velocity exceeds 2.5x baseline silence noise)
baseline_noise = motion_df.loc[time_sec < 3.5, "mean_vel"].mean()
noise_std = motion_df.loc[time_sec < 3.5, "mean_vel"].std()
thresh = baseline_noise + 3.0 * noise_std
pre_speech_mask = (time_sec >= 3.5) & (time_sec <= 4.39)
visual_onset_idx = np.where(pre_speech_mask & (motion_df["mean_vel"] > thresh))[0]
if len(visual_onset_idx) > 0:
    vis_onset_t = time_sec[visual_onset_idx[0]]
    lead_time_ms = (speech_start - vis_onset_t) * 1000.0
    ax.annotate(f"Visual Onset: {vis_onset_t:.2f}s\n(Lead: {lead_time_ms:.0f} ms before audio!)",
                xy=(vis_onset_t, motion_df.loc[visual_onset_idx[0], "mean_vel"]),
                xytext=(vis_onset_t - 1.2, ax.get_ylim()[1] * 0.75),
                arrowprops=dict(facecolor="red", shrink=0.05, width=1.5, headwidth=6),
                fontweight="bold", color="red",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor="red"))

# Subplot 2: Optical Flow Motion Energy
ax = axes[1]
ax.plot(time_sec, flow_df["motion_energy"], color="#E91E63", linewidth=2.0, label="Farneback Optical Flow Energy")
ax.set_ylabel("Motion Energy", fontweight="bold")
ax.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
ax.set_title("Pixel-Level Kinetic Energy Alignment with Assamese Word Intervals", fontsize=11, fontweight="bold")
add_segment_shading(ax, show_labels=False)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left")

fig.tight_layout()
fig.savefig(str(graphs_dir / "linguistic_kinematic_dynamics.png"), dpi=200)
plt.close(fig)
print("[OK] Saved linguistic_kinematic_dynamics.png")

# ====================================================================
# Compute Quantitative Linguistic Stats Per Word
# ====================================================================
results = []
for seg in segments[1:4]: # moi, doi, khalu
    mask = (time_sec >= seg["start"]) & (time_sec <= seg["end"])
    sub_shape = shape_df.loc[mask]
    sub_motion = motion_df.loc[mask]
    sub_flow = flow_df.loc[mask]
    
    results.append({
        "word": seg["word"],
        "assamese": seg["assamese"],
        "start_sec": seg["start"],
        "end_sec": seg["end"],
        "duration_sec": round(seg["end"] - seg["start"], 3),
        "mean_opening_px": round(float(sub_shape["mouth_opening"].mean()), 2),
        "max_opening_px": round(float(sub_shape["mouth_opening"].max()), 2),
        "mean_inner_area_px2": round(float(sub_shape["inner_area"].mean()), 2),
        "max_inner_area_px2": round(float(sub_shape["inner_area"].max()), 2),
        "mean_width_px": round(float(sub_shape["mouth_width"].mean()), 2),
        "mean_roundness": round(float(sub_shape["roundness"].mean()), 4),
        "mean_velocity_px_s": round(float(sub_motion["mean_vel"].mean()), 2),
        "peak_velocity_px_s": round(float(sub_motion["max_vel"].max()), 2),
        "mean_motion_energy": round(float(sub_flow["motion_energy"].mean()), 2),
    })

res_df = pd.DataFrame(results)
res_df.to_csv(csv_dir / "linguistic_segment_analysis.csv", index=False)
print("\n=== LINGUISTIC SEGMENT SUMMARY ===")
print(res_df.to_string())

# Save summary metadata
summary_meta = {
    "sentence": "moi doi khalu (মই দৈ খালোঁ)",
    "speaker": "s320",
    "utterance": "u1",
    "video_frames": n_frames,
    "fps": fps,
    "total_duration_sec": round(float(time_sec[-1]), 3),
    "acoustic_speech_window": [speech_start, speech_end],
    "visual_preparatory_lead_time_ms": round(lead_time_ms, 1) if 'lead_time_ms' in locals() else None,
    "segments": results
}
with open(out_dir / "linguistic_analysis_report.json", "w", encoding="utf-8") as f:
    json.dump(summary_meta, f, indent=2, ensure_ascii=False)
print("\n[OK] Analysis report saved to linguistic_analysis_report.json")

