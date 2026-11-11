"""
Linguistic Boundary & Consonant-Vowel (C-V) Segmentation Model
=============================================================
Automated model to detect:
1. Speech vs Silence Boundaries (Visual Voice Activity Detection & Preparatory Lead Time)
2. Syllable Boundaries via Aperture Velocity Zero-Crossings (d(Aperture)/dt = 0)
3. Consonant Constriction Nadirs (Valleys) vs Vowel Nuclei Expansions (Peaks)
4. Viseme Classification (Bilabials, Open Vowels, Spread Vowels, Rounded Vowels)
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt


class LinguisticSegmenter:
    """Automated linguistic and phonological feature segmenter for LMR."""

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
        lab_file: Optional[Union[str, Path]] = None,
        save_plots: bool = True
    ) -> Dict:
        """Run linguistic C-V and boundary analysis on a processed video folder."""
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

        # Load optional metadata
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

        # 3. Detect Baseline Silence & Active Speech Window (Visual VAD)
        # Baseline estimated from the first 20% or 0.8s of the recording
        silence_cutoff = min(0.8, time_sec[-1] * 0.2)
        silence_mask = time_sec < silence_cutoff
        base_aperture = np.median(smooth_aperture[silence_mask]) if np.any(silence_mask) else 2.0
        std_aperture = np.std(smooth_aperture[silence_mask]) if np.any(silence_mask) else 1.0

        aperture_thresh = base_aperture + max(2.5, 2.0 * std_aperture)

        # 4. Optional: Parse ground truth .lab if available
        lab_segments = []
        acoustic_speech_window = None
        if lab_file and Path(lab_file).exists():
            lab_segments = self._parse_lab(Path(lab_file))
            speech_words = [s for s in lab_segments if s["word"].lower() not in ("sil", "sp", "silence")]
            if speech_words:
                acoustic_speech_window = (speech_words[0]["start"], speech_words[-1]["end"])

        # 5. Detect Syllable & Boundary Inflection Points
        # Vowel Peaks: local maxima in aperture (d_aperture crossings from + to -)
        min_dist_frames = max(3, int(fps * (self.min_vowel_ms / 1000.0)))
        peak_indices, _ = find_peaks(smooth_aperture, distance=min_dist_frames, prominence=1.5)

        # Consonant Constriction Nadirs: local minima in aperture
        nadir_indices, _ = find_peaks(-smooth_aperture, distance=min_dist_frames, prominence=1.0)

        # Zero-crossings of velocity (inflection points between closing & opening)
        zero_crossings = np.where(np.diff(np.sign(d_aperture)))[0]

        # 6. Classify Continuous Frames into C-V & Viseme Classes
        frame_labels = []
        for i in range(n_frames):
            t = time_sec[i]
            ap = smooth_aperture[i]
            vel = abs(d_aperture[i])
            ar = raw_area[i]
            rnd = raw_roundness[i]
            wd = raw_width[i]

            # In silence period?
            if acoustic_speech_window:
                is_in_speech = (t >= acoustic_speech_window[0] - 0.2) and (t <= acoustic_speech_window[1] + 0.1)
            else:
                is_in_speech = ap > (base_aperture + 1.5) or vel > 30.0

            if not is_in_speech:
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

        # 7. Aggregate Contiguous Segments
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

        # 8. Measure Visual Preparatory Lead Time (Coarticulation)
        lead_time_ms = None
        if acoustic_speech_window:
            t_audio_start = acoustic_speech_window[0]
            # Search 500ms pre-speech window for first significant aperture or velocity rise
            pre_mask = (time_sec >= t_audio_start - 0.5) & (time_sec <= t_audio_start)
            pre_idx = np.where(pre_mask & (d_aperture > 15.0))[0]
            if len(pre_idx) > 0:
                t_vis_start = time_sec[pre_idx[0]]
                lead_time_ms = round(float((t_audio_start - t_vis_start) * 1000.0), 1)

        # 9. Summary Statistics
        vowels_only = seg_df[seg_df["type"] == "Vowel"]
        consonants_only = seg_df[seg_df["type"] == "Consonant"]

        summary = {
            "results_dir": str(results_dir),
            "video_file": metadata.get("video_file", "unknown"),
            "total_duration_sec": round(float(time_sec[-1]), 3),
            "total_frames": n_frames,
            "fps": round(fps, 2),
            "acoustic_speech_window": list(acoustic_speech_window) if acoustic_speech_window else None,
            "visual_lead_time_ms": lead_time_ms,
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

        # 10. Generate Automated Boundary Plot
        if save_plots:
            self._generate_plot(time_sec, raw_aperture, smooth_aperture, d_aperture,
                                peak_indices, nadir_indices, seg_df, graphs_dir,
                                lead_time_ms, acoustic_speech_window)

        return summary

    def _parse_lab(self, lab_path: Path) -> List[Dict]:
        """Parse HTK / Audacity style .lab file."""
        segments = []
        with open(lab_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        t0 = float(parts[0])
                        t1 = float(parts[1])
                        # Handle potential microsecond formatting (HTK 100ns units)
                        if t0 > 10000.0:
                            t0 /= 1e7
                            t1 /= 1e7
                        word = parts[2]
                        segments.append({"start": t0, "end": t1, "word": word})
                    except ValueError:
                        continue
        return segments

    def _generate_plot(
        self,
        time_sec: np.ndarray,
        raw_ap: np.ndarray,
        smooth_ap: np.ndarray,
        d_ap: np.ndarray,
        peaks: np.ndarray,
        nadirs: np.ndarray,
        seg_df: pd.DataFrame,
        graphs_dir: Path,
        lead_time_ms: Optional[float],
        acoustic_window: Optional[Tuple[float, float]]
    ):
        """Generate multi-panel publication plot of linguistic segmentation."""
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

        if acoustic_window:
            ax.axvline(acoustic_window[0], color="#000000", linestyle="--", linewidth=1.5, label="Acoustic Onset")
            ax.axvline(acoustic_window[1], color="#000000", linestyle="--", linewidth=1.5)

        if lead_time_ms is not None and acoustic_window:
            t_on = acoustic_window[0]
            ax.annotate(f"Visual Lead: {lead_time_ms:.0f} ms",
                        xy=(t_on, smooth_ap.max() * 0.7),
                        xytext=(t_on - 0.6, smooth_ap.max() * 0.85),
                        arrowprops=dict(facecolor="black", shrink=0.05, width=1.2, headwidth=5),
                        fontweight="bold", color="#B71C1C",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor="#D32F2F"))

        ax.set_ylabel("Mouth Opening (px)", fontweight="bold")
        ax.set_title("Automated Linguistic Segmentation: Vowel Expansions (Blue) vs Consonant Constrictions (Red)",
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

