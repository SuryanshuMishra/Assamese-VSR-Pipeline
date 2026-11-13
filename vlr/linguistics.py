"""
Video-Only Linguistic Boundary & Consonant-Vowel (C-V) Segmentation Model
========================================================================
Pure visual articulatory analysis without any audio or transcripts.
Detects:
1. Visual Speech vs Resting Silence (Visual Voice Activity Detection)
2. Syllable Transitions via Aperture Velocity Zero-Crossings (d(Aperture)/dt = 0)
3. Consonant Constriction Nadirs (Valleys) vs Vowel Nuclei Expansions (Peaks)
4. Visual Viseme Classification (Bilabials, Open Vowels, Spread Vowels, Rounded Vowels)
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt


class LinguisticSegmenter:
    """Automated vision-only linguistic feature segmenter for Lip Motion Recognition."""

    def __init__(
        self,
        min_vowel_duration_ms: float = 80.0,
        min_consonant_duration_ms: float = 50.0,
        aperture_smoothing_window: int = 7,
        polynomial_order: int = 2
    ):
        self.min_vowel_ms = min_vowel_duration_ms
        self.min_cons_ms = min_consonant_duration_ms
        self.smooth_window = aperture_smoothing_window
        self.poly_order = polynomial_order

    def process_sentence(
        self,
        results_dir: Union[str, Path],
        save_plots: bool = True
    ) -> Dict:
        """Run purely visual linguistic C-V and boundary analysis on a processed video folder."""
        results_dir = Path(results_dir)
        csv_dir = results_dir / "csvs"
        graphs_dir = results_dir / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load feature files
        shape_df = pd.read_csv(csv_dir / "shape_features.csv")
        motion_df = pd.read_csv(csv_dir / "motion_features.csv")
        lm_df = pd.read_csv(csv_dir / "landmarks.csv")
        flow_df = pd.read_csv(csv_dir / "optical_flow_features.csv")

        time_sec = lm_df["timestamp_sec"].values
        n_frames = len(time_sec)
        fps = float(1.0 / np.median(np.diff(time_sec))) if n_frames > 1 else 25.0

        # Load metadata
        meta_path = results_dir / "metadata.json"
        metadata = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        # 2. Extract and smooth key kinematic signals
        raw_aperture = shape_df["mouth_opening"].values
        raw_area = shape_df["inner_area"].values
        raw_width = shape_df["mouth_width"].values
        raw_roundness = shape_df["roundness"].values
        mean_vel = motion_df["mean_vel"].values

        # Apply Savitzky-Golay smoothing for robust derivative calculation
        w = min(self.smooth_window, n_frames if n_frames % 2 != 0 else n_frames - 1)
        if w >= 5:
            smooth_aperture = savgol_filter(raw_aperture, window_length=w, polyorder=self.poly_order)
            d_aperture = np.gradient(smooth_aperture, time_sec)
        else:
            smooth_aperture = raw_aperture
            d_aperture = shape_df["d_mouth_opening"].values

        # 3. Visual Voice Activity Detection (V-VAD): Resting Silence Baseline
        silence_cutoff = min(0.8, time_sec[-1] * 0.2)
        silence_mask = time_sec < silence_cutoff
        base_aperture = np.median(smooth_aperture[silence_mask]) if np.any(silence_mask) else 2.0
        std_aperture = np.std(smooth_aperture[silence_mask]) if np.any(silence_mask) else 1.0

        aperture_thresh = base_aperture + max(2.5, 2.0 * std_aperture)

        # 4. Detect Syllable & Boundary Inflection Points
        min_dist_frames = max(3, int(fps * (self.min_vowel_ms / 1000.0)))
        peak_indices, _ = find_peaks(smooth_aperture, distance=min_dist_frames, prominence=1.5)
        nadir_indices, _ = find_peaks(-smooth_aperture, distance=min_dist_frames, prominence=1.0)
        zero_crossings = np.where(np.diff(np.sign(d_aperture)))[0]

        # 5. Classify Frames into C-V and Viseme Classes purely from video kinematics
        frame_labels = []
        for i in range(n_frames):
            t = time_sec[i]
            ap = smooth_aperture[i]
            vel = abs(d_aperture[i])
            ar = raw_area[i]
            rnd = raw_roundness[i]
            wd = raw_width[i]

            # Visual activity check
            is_active = (ap > (base_aperture + 1.2)) or (vel > 25.0) or (ar > 150.0)

            if not is_active:
                label = "Silence"
                viseme = "RESTING"
            else:
                # Aperture expansion vs constriction
                if ap >= aperture_thresh or (ar > 300.0 and ap > base_aperture + 2.0):
                    label = "Vowel"
                    if rnd > 0.35:
                        viseme = "VOWEL_ROUNDED" # /u, o/
                    elif wd > np.percentile(raw_width, 75):
                        viseme = "VOWEL_SPREAD"  # /i, e, oi/
                    else:
                        viseme = "VOWEL_OPEN"    # /a, ɔ/
                else:
                    label = "Consonant"
                    if ap <= (base_aperture + 1.0) and ar < 100.0:
                        viseme = "BILABIAL_CLOSURE" # /m, p, b/
                    elif wd > np.percentile(raw_width, 80):
                        viseme = "DENTAL_SPREAD"    # /s, x, d/
                    else:
                        viseme = "LINGUAL_CONSTRICTION" # /t, k, l/

            frame_labels.append({
                "frame": int(lm_df.loc[i, "frame"]),
                "timestamp_sec": round(float(t), 4),
                "label": label,
                "viseme": viseme,
                "aperture_px": round(float(ap), 2),
                "inner_area_px2": round(float(ar), 2),
                "width_px": round(float(wd), 2),
                "roundness": round(float(rnd), 4),
                "d_aperture_dt": round(float(d_aperture[i]), 2)
            })

        frame_df = pd.DataFrame(frame_labels)
        frame_df.to_csv(csv_dir / "frame_linguistic_labels.csv", index=False)

        # 6. Aggregate Contiguous Segments
        segments = []
        if not frame_df.empty:
            curr_label = frame_df.loc[0, "label"]
            curr_viseme = frame_df.loc[0, "viseme"]
            start_idx = 0

            for i in range(1, len(frame_df)):
                lbl = frame_df.loc[i, "label"]
                if lbl != curr_label or i == len(frame_df) - 1:
                    end_idx = i if lbl != curr_label else i + 1
                    sub = frame_df.iloc[start_idx:end_idx]
                    t_start = sub["timestamp_sec"].iloc[0]
                    t_end = sub["timestamp_sec"].iloc[-1]
                    dur_ms = (t_end - t_start) * 1000.0

                    segments.append({
                        "segment_id": len(segments) + 1,
                        "type": curr_label,
                        "viseme": curr_viseme,
                        "start_sec": t_start,
                        "end_sec": t_end,
                        "duration_ms": round(dur_ms, 1),
                        "mean_aperture_px": round(float(sub["aperture_px"].mean()), 2),
                        "max_aperture_px": round(float(sub["aperture_px"].max()), 2),
                        "mean_inner_area_px2": round(float(sub["inner_area_px2"].mean()), 2),
                        "max_inner_area_px2": round(float(sub["inner_area_px2"].max()), 2),
                        "mean_width_px": round(float(sub["width_px"].mean()), 2),
                        "mean_roundness": round(float(sub["roundness"].mean()), 4),
                    })
                    curr_label = lbl
                    curr_viseme = frame_df.loc[i, "viseme"]
                    start_idx = i

        seg_df = pd.DataFrame(segments)
        seg_df.to_csv(csv_dir / "cv_segments.csv", index=False)

        # 7. Summary Statistics
        vowels_only = seg_df[seg_df["type"] == "Vowel"]
        consonants_only = seg_df[seg_df["type"] == "Consonant"]

        summary = {
            "results_dir": str(results_dir),
            "video_file": metadata.get("video_file", "unknown"),
            "total_duration_sec": round(float(time_sec[-1]), 3),
            "total_frames": n_frames,
            "fps": round(fps, 2),
            "vowel_segments_count": int(len(vowels_only)),
            "consonant_segments_count": int(len(consonants_only)),
            "mean_vowel_duration_ms": round(float(vowels_only["duration_ms"].mean()), 1) if not vowels_only.empty else 0.0,
            "mean_consonant_duration_ms": round(float(consonants_only["duration_ms"].mean()), 1) if not consonants_only.empty else 0.0,
            "mean_vowel_aperture_px": round(float(vowels_only["mean_aperture_px"].mean()), 2) if not vowels_only.empty else 0.0,
            "mean_consonant_aperture_px": round(float(consonants_only["mean_aperture_px"].mean()), 2) if not consonants_only.empty else 0.0,
            "peak_sentence_aperture_px": round(float(raw_aperture.max()), 2),
            "peak_sentence_inner_area_px2": round(float(raw_area.max()), 2)
        }

        with open(results_dir / "linguistic_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # 8. Generate Automated Boundary Plot (Video-Only)
        if save_plots:
            self._generate_plot(time_sec, raw_aperture, smooth_aperture, d_aperture,
                                peak_indices, nadir_indices, seg_df, graphs_dir)

        return summary

    def _generate_plot(
        self,
        time_sec: np.ndarray,
        raw_ap: np.ndarray,
        smooth_ap: np.ndarray,
        d_ap: np.ndarray,
        peaks: np.ndarray,
        nadirs: np.ndarray,
        seg_df: pd.DataFrame,
        graphs_dir: Path
    ):
        """Generate multi-panel plot of video-only linguistic segmentation."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        color_map = {
            "Vowel": "#BBDEFB",      # Light Blue
            "Consonant": "#FFCDD2",  # Light Red
            "Silence": "#F5F5F5"     # Light Grey
        }

        # Subplot 1: Aperture with C vs V segmentation shading
        ax = axes[0]
        for _, row in seg_df.iterrows():
            c = color_map.get(row["type"], "#FFFFFF")
            ax.axvspan(row["start_sec"], row["end_sec"], color=c, alpha=0.6, zorder=0)

        ax.plot(time_sec, raw_ap, color="#B0BEC5", linewidth=1.0, alpha=0.7, label="Raw Aperture (px)")
        ax.plot(time_sec, smooth_ap, color="#D32F2F", linewidth=2.0, label="Smoothed Opening (px)")

        # Mark Vowel Peaks and Consonant Valleys
        if len(peaks) > 0:
            ax.scatter(time_sec[peaks], smooth_ap[peaks], color="#1976D2", s=50, zorder=5, label="Vowel Peaks")
        if len(nadirs) > 0:
            ax.scatter(time_sec[nadirs], smooth_ap[nadirs], color="#D32F2F", marker="v", s=50, zorder=5, label="Consonant Nadirs")

        ax.set_ylabel("Mouth Opening (px)", fontweight="bold")
        ax.set_title("Video-Only Linguistic Segmentation: Vowel Expansions (Blue) vs Consonant Constrictions (Red)",
                     fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)

        # Subplot 2: Derivative Zero-Crossings & Velocity
        ax = axes[1]
        ax.plot(time_sec, d_ap, color="#E65100", linewidth=1.5, label="dAperture/dt (px/s)")
        ax.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.7)

        for _, row in seg_df.iterrows():
            c = color_map.get(row["type"], "#FFFFFF")
            ax.axvspan(row["start_sec"], row["end_sec"], color=c, alpha=0.4, zorder=0)

        ax.set_ylabel("Velocity (px/s)", fontweight="bold")
        ax.set_xlabel("Time (seconds)", fontweight="bold")
        ax.set_title("Articulatory Velocity & Syllable Transition Zero-Crossings", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)

        fig.tight_layout()
        fig.savefig(str(graphs_dir / "automated_linguistic_boundaries.png"), dpi=200)
        plt.close(fig)
