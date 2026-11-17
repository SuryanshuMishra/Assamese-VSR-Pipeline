"""
Unseen Video Inference Script for Lip Motion Recognition (LMR)
==============================================================
Runs the trained multi-speaker ML model on ANY unseen video.
Predicts Vowels vs Consonants frame-by-frame from pure video kinematics
(zero transcripts or audio required).
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from lmr_pipeline import process_video

MODEL_PATH = Path("models/best_multi_speaker_lmr_model.joblib")


def predict_unseen_video(video_path: str, output_dir: str = "unseen_prediction_output", generate_preview: bool = False):
    start_time = time.time()
    v_path = Path(video_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not v_path.exists():
        raise FileNotFoundError(f"Video file not found: {v_path}")
        
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Train model first.")
        
    print(f"\n{'='*70}")
    print(f"  Predicting Vowels & Consonants for Unseen Video")
    print(f"  Target Video: {v_path.name}")
    print(f"{'='*70}")
    
    # Step 1: Feature Extraction
    print("\n[*] Step 1: Extracting lip kinematics from video frames...")
    process_video(
        video_path=str(v_path),
        output_dir=str(out_dir),
        crop_size=96,
        generate_preview=generate_preview,
        save_debug=False
    )
    
    # Step 2: Load features
    csv_dir = out_dir / "csvs"
    shape_df = pd.read_csv(csv_dir / "shape_features.csv")
    motion_df = pd.read_csv(csv_dir / "motion_features.csv")
    lm_df = pd.read_csv(csv_dir / "landmarks.csv")
    flow_df = pd.read_csv(csv_dir / "optical_flow_features.csv")
    
    # Align with valid detected frames
    valid_lm = lm_df[lm_df["detected"] == True].reset_index(drop=True)
    n_frames = min(len(valid_lm), len(shape_df))
    if n_frames == 0:
        print("[!] No faces were detected in this video to classify.")
        return pd.DataFrame()
        
    time_sec = valid_lm["timestamp_sec"].values[:n_frames]
    frame_ids = valid_lm["frame"].values[:n_frames]
    
    # Resting width W0 for scale-invariance
    w0 = shape_df["mouth_width"].iloc[:n_frames].median()
    w0 = max(w0, 50.0)
    
    # Step 3: Load trained ML model bundle
    print(f"\n[*] Step 2: Loading trained multi-speaker model from {MODEL_PATH}...")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_names"]
    print(f"  Model type: {type(model).__name__}")
    print(f"  Trained on speakers: {bundle.get('target_speakers', 'Multi-speaker')}")
    
    # Construct feature matrix
    X_df = pd.DataFrame({
        "mouth_opening": shape_df["mouth_opening"].iloc[:n_frames],
        "inner_area": shape_df["inner_area"].iloc[:n_frames],
        "mouth_width": shape_df["mouth_width"].iloc[:n_frames],
        "roundness": shape_df["roundness"].iloc[:n_frames],
        "aspect_ratio": shape_df["aspect_ratio"].iloc[:n_frames],
        "d_mouth_opening": shape_df["mouth_opening"].iloc[:n_frames].diff().fillna(0),
        "d2_mouth_opening": shape_df["mouth_opening"].iloc[:n_frames].diff().diff().fillna(0),
        "mean_vel": motion_df["mean_vel"].iloc[:n_frames],
        "max_vel": motion_df["max_vel"].iloc[:n_frames],
        "motion_energy": flow_df["motion_energy"].iloc[:n_frames] if "motion_energy" in flow_df.columns else np.zeros(n_frames),
        "norm_aperture": shape_df["mouth_opening"].iloc[:n_frames] / w0,
        "norm_inner_area": shape_df["inner_area"].iloc[:n_frames] / (w0 ** 2),
        "norm_width": shape_df["mouth_width"].iloc[:n_frames] / w0
    })[feature_cols]
    
    # Step 4: ML Inference
    print("\n[*] Step 3: Running frame-by-frame ML inference...")
    preds = model.predict(X_df.values)
    probs = model.predict_proba(X_df.values)
    
    results = []
    for i in range(n_frames):
        c_name = "Vowel" if preds[i] == 1 else "Consonant"
        results.append({
            "frame": int(frame_ids[i]),
            "timestamp_sec": round(float(time_sec[i]), 4),
            "predicted_class": c_name,
            "label_id": int(preds[i]),
            "prob_consonant": round(float(probs[i][0]), 4),
            "prob_vowel": round(float(probs[i][1]), 4),
            "mouth_opening_px": round(float(shape_df.loc[i, "mouth_opening"]), 2),
            "inner_area_px2": round(float(shape_df.loc[i, "inner_area"]), 2),
            "roundness": round(float(shape_df.loc[i, "roundness"]), 4),
            "mean_velocity": round(float(motion_df.loc[i, "mean_vel"]), 2)
        })
        
    pred_df = pd.DataFrame(results)
    pred_csv = csv_dir / "ml_vowel_consonant_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)
    print(f"[OK] Saved frame predictions to: {pred_csv}")
    
    vowel_count = int(sum(preds == 1))
    consonant_count = int(sum(preds == 0))
    print(f"\n  Inference Summary for {v_path.name}:")
    print(f"  --------------------------------------------------")
    print(f"  Total Frames Evaluated : {n_frames}")
    print(f"  Vowels Predicted       : {vowel_count} ({vowel_count/n_frames*100:.1f}%)")
    print(f"  Consonants Predicted   : {consonant_count} ({consonant_count/n_frames*100:.1f}%)")
    print(f"  Average Vowel Conf.    : {np.mean(probs[:, 1])*100:.1f}%")
    
    # Step 5: Generate Visual Output Plot
    print("\n[*] Step 4: Generating linguistic prediction timeline plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    
    # Top plot: Aperture curve with colored regions
    ap = shape_df["mouth_opening"].values
    ax1.plot(time_sec, ap, color="#212121", linewidth=2.0, label="Lip Aperture (px)")
    
    # Highlight regions
    for i in range(len(preds)):
        t_start = time_sec[i]
        t_end = time_sec[i+1] if i + 1 < len(time_sec) else t_start + 0.033
        if preds[i] == 1:
            ax1.axvspan(t_start, t_end, color="#4CAF50", alpha=0.35, label="Predicted Vowel" if i == 0 else "")
        else:
            ax1.axvspan(t_start, t_end, color="#FF9800", alpha=0.25, label="Predicted Consonant" if i == 0 else "")
            
    ax1.set_ylabel("Mouth Opening (px)", fontsize=11, fontweight="bold")
    ax1.set_title(f"Unseen Video LMR Prediction: {v_path.name}\n(Green: Vowel Peaks, Orange: Consonant Closures)", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    
    # Remove duplicate legend entries
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc="upper right")
    
    # Bottom plot: Vowel vs Consonant Probability
    ax2.plot(time_sec, probs[:, 1], color="#2E7D32", linewidth=2.0, label="P(Vowel)")
    ax2.plot(time_sec, probs[:, 0], color="#E65100", linewidth=1.5, linestyle="--", label="P(Consonant)")
    ax2.axhline(0.5, color="gray", linestyle=":", alpha=0.7)
    ax2.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Class Probability", fontsize=11, fontweight="bold")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")
    
    fig.tight_layout()
    plot_path = out_dir / "graphs" / "ml_prediction_timeline.png"
    fig.savefig(str(plot_path), dpi=200)
    plt.close(fig)
    print(f"[OK] Timeline plot saved to: {plot_path}")
    
    elapsed = time.time() - start_time
    print(f"\n[DONE] Unseen inference completed in {elapsed:.2f} seconds.")
    print(f"{'='*70}\n")
    return pred_df


def main():
    parser = argparse.ArgumentParser(description="Predict Vowels vs Consonants on unseen video.")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", default="unseen_prediction_output", help="Output directory")
    parser.add_argument("--preview", action="store_true", help="Generate preview video with overlays")
    args = parser.parse_args()
    
    predict_unseen_video(args.video, args.output, args.preview)


if __name__ == "__main__":
    main()
