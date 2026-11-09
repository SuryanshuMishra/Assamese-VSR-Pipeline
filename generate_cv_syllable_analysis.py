"""
Consonant-Vowel (C-V) and Syllabic Linguistic Analysis for Assamese VSR
========================================================================
Decomposes the Assamese sentence "moi doi khalu" (মই দৈ খালোঁ) into:
1. Syllables: [moi] (মই), [doi] (দৈ), [kha] (খা), [lu] (লোঁ)
2. Phonetic Segments: Consonants (/m/, /d/, /kʰ/, /l/) vs Vowels/Diphthongs (/ɔi/, /a/, /u/)
Correlates kinematic parameters (aperture, inner area, spreading width, roundness, velocity)
to discover viseme boundary transitions between consonants and vowels.
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Force utf-8 stdout
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

plt.rcParams['font.sans-serif'] = ['Nirmala UI', 'Segoe UI', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

out_dir = Path("s320_1351_output")
csv_dir = out_dir / "csvs"
graphs_dir = out_dir / "graphs"
graphs_dir.mkdir(parents=True, exist_ok=True)

# Load data
shape_df = pd.read_csv(csv_dir / "shape_features.csv")
motion_df = pd.read_csv(csv_dir / "motion_features.csv")
lm_df = pd.read_csv(csv_dir / "landmarks.csv")
flow_df = pd.read_csv(csv_dir / "optical_flow_features.csv")

time_sec = lm_df["timestamp_sec"].values
n_frames = len(time_sec)

# Phonetic / Syllabic Segments for "moi doi khalu"
# Word 1: moi (4.39 - 5.00s) -> Consonant /m/ (closure) + Diphthong /ɔi/ (release & glide)
# Word 2: doi (5.00 - 5.65s) -> Consonant /d/ (dental closure) + Diphthong /ɔi/ (open-spread glide)
# Word 3: khalu (5.65 - 6.353s) -> Syllable 1 "kha" [/kʰ/ onset + /a/ open vowel]
#                                -> Syllable 2 "lu"  [/l/ transition + /u/ rounded vowel]
segments_cv = [
    {"name": "Pre-speech Silence", "ipa": "sil", "type": "Silence", "syllable": "-", "word": "sil", "start": 0.000, "end": 4.390, "color": "#F5F5F5"},
    {"name": "Consonant /m/", "ipa": "/m/", "type": "Consonant", "manner": "Bilabial Nasal", "syllable": "moi", "word": "moi", "start": 4.390, "end": 4.650, "color": "#FFCDD2"},
    {"name": "Vowel /oi/", "ipa": "/oi/", "type": "Vowel", "manner": "Open-Mid Diphthong", "syllable": "moi", "word": "moi", "start": 4.650, "end": 5.000, "color": "#BBDEFB"},
    {"name": "Consonant /d/", "ipa": "/d/", "type": "Consonant", "manner": "Dental Plosive", "syllable": "doi", "word": "doi", "start": 5.000, "end": 5.250, "color": "#FFE0B2"},
    {"name": "Vowel /oi/", "ipa": "/oi/", "type": "Vowel", "manner": "Open-Mid Diphthong", "syllable": "doi", "word": "doi", "start": 5.250, "end": 5.650, "color": "#C8E6C9"},
    {"name": "Consonant /kh/", "ipa": "/kh/", "type": "Consonant", "manner": "Velar Aspirate", "syllable": "kha", "word": "khalu", "start": 5.650, "end": 5.850, "color": "#E1BEE7"},
    {"name": "Vowel /a/", "ipa": "/a/", "type": "Vowel", "manner": "Open Central Vowel", "syllable": "kha", "word": "khalu", "start": 5.850, "end": 6.050, "color": "#B2EBF2"},
    {"name": "Consonant /l/", "ipa": "/l/", "type": "Consonant", "manner": "Alveolar Lateral", "syllable": "lu", "word": "khalu", "start": 6.050, "end": 6.150, "color": "#D1C4E9"},
    {"name": "Vowel /u/", "ipa": "/u/", "type": "Vowel", "manner": "Close Back Rounded", "syllable": "lu", "word": "khalu", "start": 6.150, "end": 6.353, "color": "#DCEDC8"},
    {"name": "Post-speech Silence", "ipa": "sil", "type": "Silence", "syllable": "-", "word": "sil", "start": 6.353, "end": 7.855, "color": "#F5F5F5"}
]

# Compute per-segment metrics
results = []
speech_segs = [s for s in segments_cv if s["type"] in ("Consonant", "Vowel")]

for seg in speech_segs:
    m = (time_sec >= seg["start"]) & (time_sec <= seg["end"])
    sub_shape = shape_df.loc[m]
    sub_motion = motion_df.loc[m]
    sub_flow = flow_df.loc[m]
    
    results.append({
        "segment": seg["name"],
        "ipa": seg["ipa"],
        "type": seg["type"],
        "manner": seg["manner"],
        "syllable": seg["syllable"],
        "word": seg["word"],
        "start_sec": seg["start"],
        "end_sec": seg["end"],
        "duration_ms": round((seg["end"] - seg["start"]) * 1000.0, 1),
        "mean_aperture_px": round(float(sub_shape["mouth_opening"].mean()), 2),
        "max_aperture_px": round(float(sub_shape["mouth_opening"].max()), 2),
        "mean_inner_area_px2": round(float(sub_shape["inner_area"].mean()), 2),
        "max_inner_area_px2": round(float(sub_shape["inner_area"].max()), 2),
        "mean_width_px": round(float(sub_shape["mouth_width"].mean()), 2),
        "mean_roundness": round(float(sub_shape["roundness"].mean()), 4),
        "mean_velocity_px_s": round(float(sub_motion["mean_vel"].mean()), 2),
        "peak_velocity_px_s": round(float(sub_motion["max_vel"].max()), 2),
        "mean_motion_energy": round(float(sub_flow["motion_energy"].mean()), 1)
    })

res_df = pd.DataFrame(results)
res_df.to_csv(csv_dir / "cv_syllable_analysis.csv", index=False)
print("=== CONSONANT VS VOWEL PHONETIC SUMMARY ===")
print(res_df[["segment", "ipa", "type", "syllable", "mean_aperture_px", "mean_inner_area_px2", "mean_width_px", "mean_roundness"]].to_string())

# ==============================================================================
# Plot: Consonant vs Vowel & Syllabic Articulatory Kinematics
# ==============================================================================
fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)

# Focus on active speech interval [4.2s to 6.6s]
t_min, t_max = 4.20, 6.55

def add_cv_shading(ax, show_labels=True):
    for seg in speech_segs:
        ax.axvspan(seg["start"], seg["end"], color=seg["color"], alpha=0.55, zorder=0)
        ax.axvline(seg["start"], color="#616161", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(speech_segs[-1]["end"], color="#616161", linestyle="--", linewidth=1.0, alpha=0.7)
    
    if show_labels:
        for seg in speech_segs:
            mid = (seg["start"] + seg["end"]) / 2.0
            tag = f"{seg['ipa']}\n({seg['type'][0]})" # C or V
            badge_color = "#D32F2F" if seg["type"] == "Consonant" else "#1976D2"
            ax.text(mid, ax.get_ylim()[1] * 0.94, tag, ha="center", va="top",
                    fontsize=9, fontweight="bold", color=badge_color,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=badge_color, alpha=0.9))

# Subplot 1: Aperture Opening & Inner Area (Primary Vowel vs Consonant Indicator)
ax = axes[0]
l1, = ax.plot(time_sec, shape_df["mouth_opening"], color="#D32F2F", linewidth=2.2, label="Vertical Aperture (Opening px)")
ax_twin = ax.twinx()
l2, = ax_twin.plot(time_sec, shape_df["inner_area"], color="#7B1FA2", linewidth=2.0, linestyle="-.", label="Inner Area (px²)")
ax.set_ylabel("Mouth Opening (px)", color="#D32F2F", fontweight="bold")
ax_twin.set_ylabel("Inner Area (px²)", color="#7B1FA2", fontweight="bold")
ax.set_title("1. Consonant Constriction vs Vowel Expansion in Assamese Speech", fontsize=13, fontweight="bold")
ax.set_xlim(t_min, t_max)
add_cv_shading(ax)
ax.grid(True, alpha=0.3)
ax.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="upper left", fontsize=9)

# Subplot 2: Horizontal Spreading vs Rounding (Viseme Discriminator)
ax = axes[1]
l1, = ax.plot(time_sec, shape_df["mouth_width"], color="#1976D2", linewidth=2.0, label="Mouth Width (Lateral Spread px)")
ax_twin = ax.twinx()
l2, = ax_twin.plot(time_sec, shape_df["roundness"], color="#388E3C", linewidth=2.0, linestyle=":", label="Roundness (Circularity)")
ax.set_ylabel("Mouth Width (px)", color="#1976D2", fontweight="bold")
ax_twin.set_ylabel("Roundness Index", color="#388E3C", fontweight="bold")
ax.set_title("2. Lateral Spreading [doi /oi/] vs Lip Rounding [khalu /u/]", fontsize=11, fontweight="bold")
ax.set_xlim(t_min, t_max)
add_cv_shading(ax, show_labels=False)
ax.grid(True, alpha=0.3)
ax.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="upper left", fontsize=9)

# Subplot 3: Aperture Derivative & Syllabic Transitions
ax = axes[2]
ax.plot(time_sec, shape_df["d_mouth_opening"], color="#E65100", linewidth=1.8, label="d(Aperture)/dt (Velocity px/s)")
ax.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.6)
ax.set_ylabel("dAperture/dt (px/s)", color="#E65100", fontweight="bold")
ax.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
ax.set_title("3. Articulatory Velocity Transitions Across Syllables [moi] -> [doi] -> [kha] -> [lu]", fontsize=11, fontweight="bold")
ax.set_xlim(t_min, t_max)
add_cv_shading(ax, show_labels=False)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", fontsize=9)

# Add syllable brackets at bottom
syllables = [
    {"label": "Syllable 1: [moi]", "start": 4.39, "end": 5.00},
    {"label": "Syllable 2: [doi]", "start": 5.00, "end": 5.65},
    {"label": "Syllable 3a: [kha]", "start": 5.65, "end": 6.05},
    {"label": "Syllable 3b: [lu]", "start": 6.05, "end": 6.353},
]
for syl in syllables:
    mid = (syl["start"] + syl["end"]) / 2.0
    ax.annotate("", xy=(syl["start"], -200), xytext=(syl["end"], -200),
                arrowprops=dict(arrowstyle="|-|", color="#212121", lw=1.5))
    ax.text(mid, -230, syl["label"], ha="center", va="top", fontsize=9, fontweight="bold", color="#212121")

fig.tight_layout()
fig.savefig(str(graphs_dir / "cv_syllable_alignment.png"), dpi=200)
plt.close(fig)
print("\n[OK] Saved cv_syllable_alignment.png")

# Aggregate Consonants vs Vowels Comparison
cv_agg = res_df.groupby("type").agg({
    "duration_ms": "mean",
    "mean_aperture_px": "mean",
    "max_aperture_px": "max",
    "mean_inner_area_px2": "mean",
    "max_inner_area_px2": "max",
    "mean_width_px": "mean",
    "mean_roundness": "mean",
    "mean_velocity_px_s": "mean"
}).round(2)
cv_agg.to_csv(csv_dir / "cv_aggregated_comparison.csv")
print("\n=== AGGREGATED CONSONANT VS VOWEL COMPARISON ===")
print(cv_agg.to_string())
