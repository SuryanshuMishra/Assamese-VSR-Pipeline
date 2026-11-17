"""
Multi-Speaker LMR Model Training & Evaluation for Vowels vs Consonants
=====================================================================
Evaluates the MAXIMUM number of speakers (4 speakers: S272, S273, S276, S278)
sharing an identical common set of 15 Assamese sentences (60 videos total).

Performs rigorous 4-Fold Leave-One-Speaker-Out (LOSO) cross-validation
and joint multi-speaker model training using strictly classical ML algorithms
(Random Forest, Gradient Boosting, SVM RBF, and Logistic Regression).
Zero neural networks used.
"""
import os
import re
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import joblib

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Pipeline Feature Extraction
from lmr_pipeline import process_video

BASE_DIR = Path("q:/IITg intern")
DATA_DIR = BASE_DIR / "cross_speaker_data"
VIDEOS_DIR = DATA_DIR / "videos"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
FEATURES_DIR = BASE_DIR / "cross_speaker_features"
MODELS_DIR = BASE_DIR / "models"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SPEAKERS = ["s272", "s273", "s276", "s278"]
TARGET_SENTENCES = [
    '1444_muk_khong_uthise',
    '1437_boroxun_ahi_ase',
    '1443_moi_sah_khalu',
    '1435_moi_football_khelu',
    '1447_azi_boroxun_ahibo',
    '1455_moi_bozar_zam',
    '1429_teu_kandi_ase',
    '1428_tai_gos_katise',
    '1445_moi_badminton_khelu',
    '1441_bola_phuri_ahu',
    '1414_dadaye_football_khelile',
    '1420_tai_porikkha_dile',
    '1453_ma_ghorot_ase',
    '1423_xi_ludu_khele',
    '1460_godhuli_phon_korilu',
]


def extract_features_for_target_videos():
    """Runs LMR feature extraction on all 60 target videos."""
    videos = []
    for spk in TARGET_SPEAKERS:
        for s in TARGET_SENTENCES:
            matches = list(VIDEOS_DIR.glob(f"{spk}_{s}_*.mp4"))
            if matches:
                videos.append(matches[0])
                
    videos = sorted(videos)
    print(f"[*] Step 1: Feature Extraction for {len(videos)} target videos across 4 speakers...")
    
    for i, v in enumerate(videos, 1):
        v_base = v.stem
        out_sub = FEATURES_DIR / v_base
        # Check if already processed
        if (out_sub / "csvs" / "shape_features.csv").exists():
            print(f"  [{i:2d}/{len(videos)}] Already processed: {v_base}")
            continue
            
        print(f"  [{i:2d}/{len(videos)}] Extracting features for: {v_base}")
        try:
            process_video(
                video_path=str(v),
                output_dir=str(out_sub),
                crop_size=96,
                generate_preview=False,
                save_debug=False
            )
        except Exception as e:
            print(f"  [!] Failed on {v_base}: {e}")
            
    print("[OK] Feature extraction complete for target videos.\n")


def parse_lab_file(lab_path: Path):
    """Parses .lab file into speech intervals."""
    words = []
    with open(lab_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    t0 = float(parts[0])
                    t1 = float(parts[1])
                    if t0 > 10000.0:
                        t0 /= 1e7
                        t1 /= 1e7
                    word = parts[2].lower()
                    words.append({"start": t0, "end": t1, "word": word})
                except ValueError:
                    continue
    return words


def build_labeled_dataset():
    """Builds ground-truth frame-level dataset of Vowels vs Consonants across 4 speakers."""
    print("[*] Step 2: Building labeled frame-by-frame dataset using transcripts...")
    all_frames = []
    
    target_bases = []
    for spk in TARGET_SPEAKERS:
        for s in TARGET_SENTENCES:
            matches = list(VIDEOS_DIR.glob(f"{spk}_{s}_*.mp4"))
            if matches:
                target_bases.append(matches[0].stem)
                
    print(f"  Target video instances to index: {len(target_bases)}")
    
    for v_base in target_bases:
        folder = FEATURES_DIR / v_base
        if not folder.exists() or not (folder / "csvs" / "shape_features.csv").exists():
            continue
            
        m = re.match(r'^(s\d+(?:\s*[fF])?)_(\d+_[a-zA-Z0-9_]+)_(u\d+)$', v_base)
        if not m:
            continue
        speaker, sentence_key, utterance = m.groups()
        speaker = speaker.strip().lower()
        
        # Load features
        shape_df = pd.read_csv(folder / "csvs" / "shape_features.csv")
        motion_df = pd.read_csv(folder / "csvs" / "motion_features.csv")
        lm_df = pd.read_csv(folder / "csvs" / "landmarks.csv")
        flow_df = pd.read_csv(folder / "csvs" / "optical_flow_features.csv")
        
        time_sec = lm_df["timestamp_sec"].values
        n_frames = len(time_sec)
        
        # Load matching .lab (match by speaker + sentence key, any utterance)
        lab_file = ANNOTATIONS_DIR / f"{v_base}.lab"
        if not lab_file.exists():
            # Try to find a lab with same speaker + sentence but different utterance tag
            lab_candidates = list(ANNOTATIONS_DIR.glob(f"{speaker}_{sentence_key}_*.lab"))
            if lab_candidates:
                lab_file = lab_candidates[0]
            else:
                continue
            
        lab_words = parse_lab_file(lab_file)
        speech_words = [w for w in lab_words if w["word"] not in ("sil", "sp", "silence")]
        if not speech_words:
            continue
            
        speech_start = speech_words[0]["start"]
        speech_end = speech_words[-1]["end"]
        
        # Resting Lip Width W0
        w0 = np.median(shape_df.loc[time_sec < min(0.6, speech_start), "mouth_width"]) if np.any(time_sec < speech_start) else 100.0
        w0 = max(w0, 50.0)
        
        # Aperture smoothing
        raw_ap = shape_df["mouth_opening"].values
        w_len = min(7, n_frames if n_frames % 2 != 0 else n_frames - 1)
        smooth_ap = savgol_filter(raw_ap, window_length=w_len, polyorder=2) if w_len >= 5 else raw_ap
        d_ap = np.gradient(smooth_ap, time_sec)
        d2_ap = np.gradient(d_ap, time_sec)
        
        # Detect peaks (Vowels) and nadirs (Consonants) during speech
        in_speech_mask = (time_sec >= speech_start) & (time_sec <= speech_end)
        peaks, _ = find_peaks(smooth_ap, distance=2, prominence=1.5)
        nadirs, _ = find_peaks(-smooth_ap, distance=2, prominence=1.5)
        
        # Median active opening threshold
        speech_ap_median = np.median(smooth_ap[in_speech_mask]) if np.any(in_speech_mask) else 10.0
        
        # Assign ground-truth labels for each frame
        for i in range(n_frames):
            t = time_sec[i]
            
            # 1. Silence check
            if t < speech_start or t > speech_end:
                label_name = "Silence"
                binary_label = -1 # Excluded from C vs V binary classification
            else:
                # Active speech frame: Vowel vs Consonant
                ap = smooth_ap[i]
                area = shape_df.loc[i, "inner_area"]
                
                dist_to_peak = min(abs(i - p) for p in peaks) if len(peaks) > 0 else 999
                dist_to_nadir = min(abs(i - n) for n in nadirs) if len(nadirs) > 0 else 999
                
                if (dist_to_peak <= 1 and ap > speech_ap_median * 0.9) or (ap >= speech_ap_median and area > 200.0):
                    label_name = "Vowel"
                    binary_label = 1
                else:
                    label_name = "Consonant"
                    binary_label = 0
                    
            all_frames.append({
                "video_base": v_base,
                "speaker": speaker,
                "sentence_key": sentence_key,
                "utterance": utterance,
                "frame": int(lm_df.loc[i, "frame"]),
                "timestamp_sec": round(float(t), 4),
                "label_name": label_name,
                "label": binary_label,
                # Physical features
                "mouth_opening": round(float(raw_ap[i]), 3),
                "smooth_aperture": round(float(smooth_ap[i]), 3),
                "inner_area": round(float(shape_df.loc[i, "inner_area"]), 2),
                "mouth_width": round(float(shape_df.loc[i, "mouth_width"]), 2),
                "roundness": round(float(shape_df.loc[i, "roundness"]), 4),
                "aspect_ratio": round(float(shape_df.loc[i, "aspect_ratio"]), 4),
                "d_mouth_opening": round(float(d_ap[i]), 3),
                "d2_mouth_opening": round(float(d2_ap[i]), 3),
                "mean_vel": round(float(motion_df.loc[i, "mean_vel"]), 3),
                "max_vel": round(float(motion_df.loc[i, "max_vel"]), 3),
                "motion_energy": round(float(flow_df.loc[i, "motion_energy"]), 1),
                # Scale-invariant normalized features
                "norm_aperture": round(float(raw_ap[i] / w0), 4),
                "norm_inner_area": round(float(shape_df.loc[i, "inner_area"] / (w0 ** 2)), 6),
                "norm_width": round(float(shape_df.loc[i, "mouth_width"] / w0), 4),
            })
            
    df = pd.DataFrame(all_frames)
    df.to_csv(BASE_DIR / "multi_speaker_dataset.csv", index=False)
    print(f"[OK] Multi-speaker dataset saved with {len(df)} total frames.")
    speech_df = df[df["label"] != -1]
    print(f"     Active Speech Frames: {len(speech_df)}")
    print(f"     Vowels (Class 1): {sum(speech_df['label'] == 1)}")
    print(f"     Consonants (Class 0): {sum(speech_df['label'] == 0)}")
    for spk in sorted(df['speaker'].unique()):
        spk_df = speech_df[speech_df['speaker'] == spk]
        print(f"       - Speaker {spk:4s}: {len(spk_df)} active frames ({sum(spk_df['label']==1)} V, {sum(spk_df['label']==0)} C)")
    print()
    return df


def train_and_evaluate_multi_speaker(df: pd.DataFrame):
    """Performs 4-Fold Leave-One-Speaker-Out (LOSO) Cross-Validation across S272, S273, S276, S278."""
    print("[*] Step 3: 4-Fold Leave-One-Speaker-Out (LOSO) Model Evaluation...")
    
    speech_df = df[df["label"] != -1].copy()
    
    feature_cols = [
        "mouth_opening", "inner_area", "mouth_width", "roundness", "aspect_ratio",
        "d_mouth_opening", "d2_mouth_opening", "mean_vel", "max_vel", "motion_energy",
        "norm_aperture", "norm_inner_area", "norm_width"
    ]
    
    # Classical ML Models (No Neural Networks)
    def get_models():
        return {
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
            "Gradient Boosting": HistGradientBoostingClassifier(max_iter=100, random_state=42),
            "Support Vector Machine (RBF)": Pipeline([
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="rbf", C=1.0, random_state=42))
            ]),
            "Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, random_state=42))
            ])
        }

    loso_results = {name: [] for name in get_models().keys()}
    fold_reports = {}
    
    print("\n" + "="*80)
    print(f"{'Fold / Held-out Speaker':<28} | {'Model':<26} | {'Acc':<7} | {'Prec':<7} | {'Rec':<7} | {'F1':<7}")
    print("="*80)
    
    for held_out in TARGET_SPEAKERS:
        train_df = speech_df[speech_df["speaker"] != held_out]
        test_df = speech_df[speech_df["speaker"] == held_out]
        
        X_train = train_df[feature_cols].values
        y_train = train_df["label"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["label"].values
        
        models = get_models()
        fold_reports[held_out] = {}
        
        for name, clf in models.items():
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            
            metrics = {
                "accuracy": round(float(acc), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1_score": round(float(f1), 4)
            }
            loso_results[name].append(metrics)
            fold_reports[held_out][name] = metrics
            
            print(f"Test on {held_out:4s} (Train on 3 other spks) | {name:<26} | {acc*100:5.2f}% | {prec*100:5.2f}% | {rec*100:5.2f}% | {f1*100:5.2f}%")
        print("-" * 80)
        
    # Summary of LOSO Averages
    print("\n" + "="*80)
    print("  OVERALL LEAVE-ONE-SPEAKER-OUT (LOSO) SUMMARY ACROSS ALL 4 SPEAKERS")
    print("="*80)
    mean_summary = {}
    for name, m_list in loso_results.items():
        avg_acc = np.mean([m["accuracy"] for m in m_list])
        std_acc = np.std([m["accuracy"] for m in m_list])
        avg_prec = np.mean([m["precision"] for m in m_list])
        avg_rec = np.mean([m["recall"] for m in m_list])
        avg_f1 = np.mean([m["f1_score"] for m in m_list])
        
        mean_summary[name] = {
            "mean_accuracy": round(float(avg_acc), 4),
            "std_accuracy": round(float(std_acc), 4),
            "mean_precision": round(float(avg_prec), 4),
            "mean_recall": round(float(avg_rec), 4),
            "mean_f1_score": round(float(avg_f1), 4)
        }
        print(f"  {name:<28} | Mean Acc: {avg_acc*100:5.2f}% ± {std_acc*100:4.2f}% | Prec: {avg_prec*100:5.2f}% | Rec: {avg_rec*100:5.2f}% | Mean F1: {avg_f1*100:5.2f}%")
    print("="*80 + "\n")
    
    # Step 4: Joint Multi-Speaker Model
    X_all = speech_df[feature_cols].values
    y_all = speech_df["label"].values
    
    best_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    best_rf.fit(X_all, y_all)
    all_preds = best_rf.predict(X_all)
    joint_acc = accuracy_score(y_all, all_preds)
    joint_f1 = f1_score(y_all, all_preds)
    print(f"[*] Joint 4-Speaker Model Fit | Acc: {joint_acc*100:5.2f}% | F1: {joint_f1*100:5.2f}%")
    
    # Save the trained model
    model_path = MODELS_DIR / "best_multi_speaker_lmr_model.joblib"
    joblib.dump({
        "model": best_rf,
        "feature_names": feature_cols,
        "classes": ["Consonant", "Vowel"],
        "target_speakers": TARGET_SPEAKERS,
        "loso_mean_metrics": mean_summary,
        "fold_reports": fold_reports
    }, model_path)
    print(f"[OK] Best 4-Speaker model saved to: {model_path}")
    
    # Plotting
    importances = best_rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Subplot 1: Feature Importances
    ax = axes[0]
    ax.barh(range(len(feature_cols)), importances[indices[::-1]], color="#1E88E5", align="center")
    ax.set_yticks(range(len(feature_cols)))
    ax.set_yticklabels([feature_cols[i] for i in indices[::-1]], fontsize=10, fontweight="bold")
    ax.set_xlabel("Feature Importance Score (Gini)", fontsize=11, fontweight="bold")
    ax.set_title("Top Multi-Speaker Lip Kinematic Predictors", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    # Subplot 2: LOSO Accuracy across Speakers
    ax = axes[1]
    model_names = list(mean_summary.keys())
    x = np.arange(len(TARGET_SPEAKERS))
    width = 0.18
    
    colors = ["#43A047", "#1E88E5", "#FB8C00", "#8E24AA"]
    for i, m_name in enumerate(model_names):
        spk_accs = [fold_reports[spk][m_name]["accuracy"] * 100 for spk in TARGET_SPEAKERS]
        ax.bar(x + i*width - width*1.5, spk_accs, width, label=m_name, color=colors[i % len(colors)])
        
    ax.set_ylabel("Held-Out Speaker Test Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("Leave-One-Speaker-Out (LOSO) Generalization across 4 Speakers", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Test on {s.upper()}" for s in TARGET_SPEAKERS], fontsize=10, fontweight="bold")
    ax.set_ylim(40, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    
    fig.tight_layout()
    fig.savefig(str(MODELS_DIR / "multi_speaker_evaluation_plots.png"), dpi=200)
    plt.close(fig)
    print(f"[OK] Evaluation plot saved to: {MODELS_DIR / 'multi_speaker_evaluation_plots.png'}")
    
    # Save JSON report
    report = {
        "speakers_count": 4,
        "speakers": TARGET_SPEAKERS,
        "sentences_count": len(TARGET_SENTENCES),
        "total_active_speech_frames": len(X_all),
        "vowels_count": int(sum(y_all == 1)),
        "consonants_count": int(sum(y_all == 0)),
        "top_predictive_features": [feature_cols[i] for i in indices[:5]],
        "loso_mean_summary": mean_summary,
        "loso_fold_breakdown": fold_reports,
        "joint_model_accuracy": round(float(joint_acc), 4)
    }
    with open(MODELS_DIR / "multi_speaker_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Detailed multi-speaker report saved to: {MODELS_DIR / 'multi_speaker_evaluation_report.json'}")


def main():
    start_time = time.time()
    extract_features_for_target_videos()
    df = build_labeled_dataset()
    train_and_evaluate_multi_speaker(df)
    elapsed = time.time() - start_time
    print(f"\n[DONE] Multi-Speaker pipeline completed in {elapsed/60.0:.2f} minutes.")


if __name__ == "__main__":
    main()
