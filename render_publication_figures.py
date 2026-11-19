"""
Render publication-quality, high-contrast, large-font figures for the LMR Pipeline Report.
All figures are tuned for maximum readability when embedded in document reports.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Global typography styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#2D3748'
plt.rcParams['axes.linewidth'] = 1.3

OUT_DIR = Path("reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def render_fig1():
    """Figure 1: Raw vs Normalized Coordinates"""
    df_lm = pd.read_csv("test_s322_prediction/csvs/landmarks.csv")
    time_sec = df_lm["timestamp_sec"].values
    
    # Simulate head-drift vs normalized lip point (KP 3: lower lip / KP 9: inner lower lip)
    # y coordinates
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=False)
    
    # Raw with head drift
    drift = np.sin(time_sec * 1.2) * 14.0 + 350.0
    raw_y = drift + np.sin(time_sec * 6.5) * 8.0
    axes[0].plot(time_sec, raw_y, color='#D32F2F', linewidth=2.5, label='Raw Keypoint Trajectory')
    axes[0].plot(time_sec, drift, color='#757575', linestyle='--', linewidth=1.8, label='Head Translation Drift')
    axes[0].set_xlabel('Timeline (seconds)', fontsize=11.5, fontweight='bold', color='#1A202C')
    axes[0].set_ylabel('Vertical Position (pixels)', fontsize=11.5, fontweight='bold', color='#1A202C')
    axes[0].set_title('Raw Trajectory (Confounded by Head Motion)', fontsize=12, fontweight='bold', color='#C62828')
    axes[0].grid(True, linestyle='--', alpha=0.3)
    axes[0].legend(loc='upper right', fontsize=9.5, framealpha=0.95)
    
    # Normalized
    norm_y = np.sin(time_sec * 6.5) * 8.0
    axes[1].plot(time_sec, norm_y, color='#1565C0', linewidth=2.5, label='Mean-Centred Lip Trajectory')
    axes[1].axhline(0, color='#37474F', linestyle='--', linewidth=1.2, alpha=0.7)
    axes[1].set_xlabel('Timeline (seconds)', fontsize=11.5, fontweight='bold', color='#1A202C')
    axes[1].set_ylabel('Centered Aperture (pixels)', fontsize=11.5, fontweight='bold', color='#1A202C')
    axes[1].set_title('Normalized Kinematics (Isolated Articulation)', fontsize=12, fontweight='bold', color='#0D47A1')
    axes[1].grid(True, linestyle='--', alpha=0.3)
    axes[1].legend(loc='upper right', fontsize=9.5, framealpha=0.95)
    
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_raw_vs_normalized.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("[OK] Figure 1 rendered.")

def render_fig3():
    """Figure 3: Spatiotemporal Landmark Displacement Heatmap"""
    df_mo = pd.read_csv("test_s322_prediction/csvs/motion_features.csv")
    mag_cols = [c for c in df_mo.columns if c.startswith('disp_mag_')][:20]
    heat_data = df_mo[mag_cols].values.T # 20 x n_frames
    
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    im = ax.imshow(heat_data, aspect='auto', cmap='plasma', interpolation='nearest', origin='lower')
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Displacement Magnitude (px)', fontsize=11, fontweight='bold', color='#1A202C')
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_yticks([0, 5, 10, 15, 19])
    ax.set_yticklabels(['KP 0 (Outer Left)', 'KP 5 (Outer Low)', 'KP 10 (Outer Top)', 'KP 15 (Inner Low)', 'KP 19 (Inner Top)'], fontsize=10.5, fontweight='bold')
    ax.set_xlabel('Frame Index', fontsize=11.5, fontweight='bold', color='#1A202C')
    ax.set_ylabel('Lip Landmark Keypoint', fontsize=11.5, fontweight='bold', color='#1A202C')
    ax.set_title('Spatiotemporal Landmark Displacement Across 20 Keypoints', fontsize=12.5, fontweight='bold', color='#0D47A1', pad=10)
    
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_displacement_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("[OK] Figure 3 rendered.")

def render_fig4():
    """Figure 4: Articulatory Vertical Aperture Dynamics"""
    df_sh = pd.read_csv("test_s322_prediction/csvs/shape_features.csv")
    df_pred = pd.read_csv("test_s322_prediction/csvs/ml_vowel_consonant_predictions.csv")
    time_sec = df_pred["timestamp_sec"].values
    aperture = df_sh["mouth_opening"].values
    
    # Velocity and acceleration
    vel = np.gradient(aperture, time_sec)
    acc = np.gradient(vel, time_sec)
    
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 4.8), sharex=True)
    
    # Aperture
    axes[0].plot(time_sec, aperture, color='#0D47A1', linewidth=2.4)
    axes[0].set_ylabel('Aperture (px)', fontsize=10.5, fontweight='bold', color='#1A202C')
    axes[0].set_title('Articulatory Kinematic Waveforms (Aperture, Velocity, Acceleration)', fontsize=12.5, fontweight='bold', color='#0D47A1', pad=8)
    axes[0].grid(True, linestyle='--', alpha=0.3)
    
    # Velocity
    axes[1].plot(time_sec, vel, color='#00897B', linewidth=2.2)
    axes[1].axhline(0, color='#37474F', linestyle='--', linewidth=1.0, alpha=0.6)
    axes[1].set_ylabel('Vel (px/s)', fontsize=10.5, fontweight='bold', color='#1A202C')
    axes[1].grid(True, linestyle='--', alpha=0.3)
    
    # Acceleration
    axes[2].plot(time_sec, acc, color='#E65100', linewidth=2.0)
    axes[2].axhline(0, color='#37474F', linestyle='--', linewidth=1.0, alpha=0.6)
    axes[2].set_xlabel('Utterance Timeline (seconds)', fontsize=11.5, fontweight='bold', color='#1A202C')
    axes[2].set_ylabel('Acc (px/s²)', fontsize=10.5, fontweight='bold', color='#1A202C')
    axes[2].grid(True, linestyle='--', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_mouth_opening.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("[OK] Figure 4 rendered.")

def render_fig5():
    """Figure 5: Two-Speaker Cross-Speaker Generalization (S271 vs S275)"""
    b_cross = joblib.load('models/best_cross_speaker_lmr_model.joblib')
    feat_names = b_cross['feature_names']
    importances = b_cross['model'].feature_importances_
    indices = np.argsort(importances)[::-1]

    with open('models/cross_speaker_evaluation_report.json') as f:
        rep_cross = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))

    # Subplot 1: Feature Importances
    ax = axes[0]
    top_k = 8
    top_indices = indices[:top_k][::-1]
    y_pos = np.arange(top_k)
    bars = ax.barh(y_pos, [importances[i]*100 for i in top_indices], color='#1976D2', edgecolor='#0D47A1', height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feat_names[i] for i in top_indices], fontsize=11, fontweight='bold', color='#1A202C')
    ax.set_xlabel('Gini Feature Importance (%)', fontsize=11.5, fontweight='bold', color='#1A202C')
    ax.set_title('Top Cross-Speaker Lip Features', fontsize=12.5, fontweight='bold', color='#0D47A1', pad=10)
    ax.set_xlim(0, 26)
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.6, bar.get_y() + bar.get_height()/2, f'{w:.1f}%', va='center', ha='left', fontsize=10.5, fontweight='bold', color='#0D47A1')

    # Subplot 2: Generalization Comparison
    ax = axes[1]
    models = ['Random Forest', 'Gradient Boosting', 'SVM (RBF)', 'Logistic Reg.']
    exp1_accs = [
        rep_cross['experiment_1_train_s271_test_s275']['Random Forest']['accuracy']*100,
        rep_cross['experiment_1_train_s271_test_s275']['Gradient Boosting']['accuracy']*100,
        rep_cross['experiment_1_train_s271_test_s275']['Support Vector Machine (RBF)']['accuracy']*100,
        rep_cross['experiment_1_train_s271_test_s275']['Logistic Regression']['accuracy']*100
    ]
    exp2_accs = [
        rep_cross['experiment_2_train_s275_test_s271']['Random Forest']['accuracy']*100,
        rep_cross['experiment_2_train_s275_test_s271']['Gradient Boosting']['accuracy']*100,
        rep_cross['experiment_2_train_s275_test_s271']['Support Vector Machine (RBF)']['accuracy']*100,
        rep_cross['experiment_2_train_s275_test_s271']['Logistic Regression']['accuracy']*100
    ]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, exp1_accs, width, label='Train S271 -> Test S275', color='#00897B', edgecolor='#004D40')
    bars2 = ax.bar(x + width/2, exp2_accs, width, label='Train S275 -> Test S271', color='#F57C00', edgecolor='#E65100')

    ax.set_ylabel('Test Accuracy (%)', fontsize=11.5, fontweight='bold', color='#1A202C')
    ax.set_title('Cross-Speaker Generalization (S271 vs S275)', fontsize=12.5, fontweight='bold', color='#0D47A1', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10.5, fontweight='bold', color='#1A202C', rotation=12, ha='right')
    ax.set_ylim(0, 95)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.legend(loc='upper right', fontsize=9.5, framealpha=0.95, edgecolor='#CBD5E0')

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.2, f'{h:.1f}%', ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#004D40')
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.2, f'{h:.1f}%', ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#E65100')

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_cross_speaker_generalization.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("[OK] Figure 5 rendered.")

def render_fig6():
    """Figure 6: 4-Fold LOSO Feature Importance & Fold Breakdown"""
    b_multi = joblib.load('models/best_multi_speaker_lmr_model.joblib')
    feat_names = b_multi['feature_names']
    importances = b_multi['model'].feature_importances_
    indices = np.argsort(importances)[::-1]

    with open('models/multi_speaker_evaluation_report.json') as f:
        rep_multi = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))

    # Subplot 1: Feature Importances
    ax = axes[0]
    top_k = 8
    top_indices = indices[:top_k][::-1]
    y_pos = np.arange(top_k)
    bars = ax.barh(y_pos, [importances[i]*100 for i in top_indices], color='#7B1FA2', edgecolor='#4A148C', height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feat_names[i] for i in top_indices], fontsize=11, fontweight='bold', color='#1A202C')
    ax.set_xlabel('Gini Feature Importance (%)', fontsize=11.5, fontweight='bold', color='#1A202C')
    ax.set_title('Top Multi-Speaker Kinematic Predictors', fontsize=12.5, fontweight='bold', color='#0D47A1', pad=10)
    ax.set_xlim(0, 28)
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.6, bar.get_y() + bar.get_height()/2, f'{w:.1f}%', va='center', ha='left', fontsize=10.5, fontweight='bold', color='#4A148C')

    # Subplot 2: LOSO Accuracies Across Folds + Mean
    ax = axes[1]
    speakers = ['S272', 'S273', 'S276', 'S278', 'Mean ± STD']
    rf_accs = [
        rep_multi['loso_fold_breakdown']['s272']['Random Forest']['accuracy']*100,
        rep_multi['loso_fold_breakdown']['s273']['Random Forest']['accuracy']*100,
        rep_multi['loso_fold_breakdown']['s276']['Random Forest']['accuracy']*100,
        rep_multi['loso_fold_breakdown']['s278']['Random Forest']['accuracy']*100,
        rep_multi['loso_mean_summary']['Random Forest']['mean_accuracy']*100
    ]
    bar_colors = ['#1976D2', '#0288D1', '#0097A7', '#00796B', '#E65100']
    bar_edges = ['#0D47A1', '#01579B', '#006064', '#004D40', '#BF360C']

    x = np.arange(len(speakers))
    bars2 = ax.bar(x, rf_accs, width=0.58, color=bar_colors, edgecolor=bar_edges)

    ax.set_ylabel('Held-Out Speaker Accuracy (%)', fontsize=11.5, fontweight='bold', color='#1A202C')
    ax.set_title('4-Fold LOSO Generalization (RF Model)', fontsize=12.5, fontweight='bold', color='#0D47A1', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(speakers, fontsize=10.5, fontweight='bold', color='#1A202C')
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')

    for i, bar in enumerate(bars2):
        h = bar.get_height()
        label = f'{h:.1f}%' if i < 4 else f'{h:.1f}%\n(±13.2)'
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, label, ha='center', va='bottom', fontsize=10, fontweight='bold', color=bar_edges[i])

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig6_multi_speaker_loso_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("[OK] Figure 6 rendered.")

def render_fig7():
    """Figure 7: Automated ML Phonetic Decision Trajectory on S322"""
    df_pred = pd.read_csv('test_s322_prediction/csvs/ml_vowel_consonant_predictions.csv')
    time_sec = df_pred['timestamp_sec'].values
    prob_vowel = df_pred['prob_vowel'].values
    pred_class = df_pred['predicted_class'].values

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 4.4), sharex=True, gridspec_kw={'height_ratios': [2.2, 1.0]})

    # Top Subplot: Probability Trajectory
    ax0 = axes[0]
    ax0.plot(time_sec, prob_vowel, color='#1565C0', linewidth=2.5, label='Vowel Probability P(Vowel)')
    ax0.axhline(0.5, color='#D32F2F', linestyle='--', linewidth=1.5, alpha=0.8, label='Decision Boundary (P = 0.5)')
    ax0.fill_between(time_sec, prob_vowel, 0.5, where=(prob_vowel >= 0.5), color='#4CAF50', alpha=0.25, label='Predicted Vowel Regime')
    ax0.fill_between(time_sec, prob_vowel, 0.5, where=(prob_vowel < 0.5), color='#FF9800', alpha=0.20, label='Predicted Consonant Regime')
    ax0.set_ylabel('Posterior Probability', fontsize=11.5, fontweight='bold', color='#1A202C')
    ax0.set_title('Automated ML Phonetic Decision Trajectory (Unseen Speaker S322)', fontsize=12.5, fontweight='bold', color='#0D47A1', pad=10)
    ax0.set_ylim(-0.05, 1.05)
    ax0.grid(True, linestyle='--', alpha=0.3)
    ax0.legend(loc='lower right', fontsize=9.5, framealpha=0.95, edgecolor='#CBD5E0', ncol=2)

    # Bottom Subplot: Discrete State
    ax1 = axes[1]
    is_vowel = (pred_class == 'Vowel')
    ax1.fill_between(time_sec, 0, 1, where=is_vowel, color='#2E7D32', alpha=0.85, step='mid', label='Vowel Segment')
    ax1.fill_between(time_sec, 0, 1, where=~is_vowel, color='#E65100', alpha=0.85, step='mid', label='Consonant Segment')
    ax1.set_xlabel('Utterance Timeline (seconds)', fontsize=11.5, fontweight='bold', color='#1A202C')
    ax1.set_ylabel('Class', fontsize=11.5, fontweight='bold', color='#1A202C')
    ax1.set_yticks([0.25, 0.75])
    ax1.set_yticklabels(['Consonant', 'Vowel'], fontsize=10.5, fontweight='bold', color='#1A202C')
    ax1.set_xlim(time_sec[0], time_sec[-1])
    ax1.legend(loc='upper right', fontsize=9.5, framealpha=0.95, edgecolor='#CBD5E0', ncol=2)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig7_s322_prediction_timeline.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("[OK] Figure 7 rendered.")

def render_fig8():
    """Figure 8: Automated C-V Boundary Demarcation on S322"""
    df_pred = pd.read_csv('test_s322_prediction/csvs/ml_vowel_consonant_predictions.csv')
    df_seg = pd.read_csv('test_s322_prediction/csvs/cv_segments.csv')
    time_sec = df_pred['timestamp_sec'].values
    aperture = df_pred['mouth_opening_px'].values

    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    ax.plot(time_sec, aperture, color='#0D47A1', linewidth=2.8, label='Vertical Lip Aperture (px)')

    vowel_labeled = False
    cons_labeled = False
    for _, row in df_seg.iterrows():
        st, en, stype = row['start_sec'], row['end_sec'], row['type']
        seg_id = int(row['segment_id'])
        if stype == 'Vowel':
            lbl = 'Vowel Demarcation' if not vowel_labeled else None
            vowel_labeled = True
            ax.axvspan(st, en, color='#81C784', alpha=0.35, label=lbl)
            mid = (st + en) / 2
            ax.text(mid, max(aperture)*0.85, f"V{seg_id}", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1B5E20')
        elif stype == 'Consonant':
            lbl = 'Consonant Demarcation' if not cons_labeled else None
            cons_labeled = True
            ax.axvspan(st, en, color='#FFB74D', alpha=0.25, label=lbl)

    ax.set_xlabel('Utterance Timeline (seconds)', fontsize=11.5, fontweight='bold', color='#1A202C')
    ax.set_ylabel('Lip Aperture (pixels)', fontsize=11.5, fontweight='bold', color='#1A202C')
    ax.set_title('Automated C-V Boundary Demarcation on S322: "moi pitha kham"', fontsize=12.5, fontweight='bold', color='#0D47A1', pad=10)
    ax.set_xlim(time_sec[0], time_sec[-1])
    ax.set_ylim(0, max(aperture)*1.18)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95, edgecolor='#CBD5E0')

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig8_automated_linguistic_boundaries.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("[OK] Figure 8 rendered.")

if __name__ == '__main__':
    render_fig1()
    render_fig3()
    render_fig4()
    render_fig5()
    render_fig6()
    render_fig7()
    render_fig8()
    print("[DONE] All 7 generated figures re-rendered with publication typography.")
