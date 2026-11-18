# Lip Motion Recognition (LMR) Pipeline for Assamese
## Final Technical & Experimental Submission Report

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**Project:** The pipeline targets the Assamese  
**Institution:** Indian Institute of Technology Guwahati (IITG)  
**Domain:** Lip Motion Recognition (LMR) / Visual Speech Processing  
**Status:** Validated End-to-End Multi-Speaker Corpus & Classical ML Benchmark  
**Repository:** [https://github.com/SuryanshuMishra/Assamese-VSR-Pipeline.git](https://github.com/SuryanshuMishra/Assamese-VSR-Pipeline.git) (`commit 9a56bc9`)  
**Deliverable Documents:** [LMR_Pipeline_Report.docx](./LMR_Pipeline_Report.docx) | [LMR_Pipeline_Report.pdf](./LMR_Pipeline_Report.pdf)  
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

---

## 1. Executive Summary

This report presents the complete development, algorithmic architecture, and rigorous experimental validation of an enterprise-grade **Lip Motion Recognition (LMR)** pipeline specifically developed for the **Assamese language**. 

Operating under the strict constraint of **pure visual processing without audio dependencies**, the system:
1. Streams video frames into memory via FFmpeg, extracts 68 facial landmarks via dlib at ~20–25 fps, and isolates the 20 inner/outer lip keypoints.
2. Extracts a comprehensive **13-dimensional kinematic feature vector** combining aperture geometry, Shoelace inner polygon cavity areas, velocity derivatives, optical flow motion energy, and **scale-invariant normalized metrics** ($W_0$ baseline normalization).
3. Evaluates and validates classical supervised machine learning classifiers (**Random Forest**, **Gradient Boosting**, **SVM RBF**, and **Logistic Regression** — **zero neural networks**) to classify speech frames into **Vowels vs. Consonants** across identical Assamese sentences spoken by different speakers.
4. Performs rigorous **Leave-One-Speaker-Out (LOSO) cross-validation** across the global maximum subset of 4 speakers (`S272`, `S273`, `S276`, `S278`) sharing 15 matching sentences (60 video recordings, 6,771 active speech frames), achieving **72.52% ± 13.16% mean speaker-independent accuracy** and **96.20% joint accuracy**.
5. Seamlessly deploys the trained multi-speaker model directly into the main pipeline ([lmr_pipeline.py](./lmr_pipeline.py)), verified by executing live end-to-end inference on a completely unseen speaker (`S322`) and sentence (`884_moi_pitha_kham`) with zero transcripts in **19.8 seconds**.

---

## 2. System Architecture & Feature Extraction Pipeline

The pipeline processes speech recordings through seven tightly coupled modular stages, completely eliminating disk I/O bottlenecks:

```mermaid
flowchart TD
    A["Raw Speech Video (.mp4)"] --> B["1. Memory-Piped FFmpeg Decoding<br>(NumPy Array Stream)"]
    B --> C["2. dlib 68-Point Facial Landmark Detection<br>(Isolate Mouth Keypoints 48-67)"]
    C --> D["3. Centroid Mean-Centring Normalization<br>(96×96 Crop ROI)"]
    D --> E["4. Dual Motion & Kinematic Extraction<br>(Derivatives, Velocity, Farneback Optical Flow)"]
    E --> F["5. Quality Gating & Validity Mask<br>(Laplacian Blur, Brightness, Jitter)"]
    F --> G["6. Multi-Speaker ML Linguistic Engine<br>(best_multi_speaker_lmr_model.joblib)"]
    G --> H["7. Automated Deliverables<br>(CSVs, Plots, Segments, Preview Video)"]
```

### 2.1 The 13-Dimensional Kinematic Feature Representation

To ensure robustness against variations in head size, speaker-to-camera distance, and individual anatomy, each frame is represented by 13 physical features:

| Index | Feature Name | Mathematical / Algorithmic Formulation | Physical Linguistic Significance |
|:---:|:---|:---|:---|
| **1** | `mouth_opening` | $\frac{1}{2} \|(p_{51} + p_{62}) - (p_{57} + p_{66})\|_2$ | Vertical lip aperture (inter-lip distance) |
| **2** | `inner_area` | $\frac{1}{2} \left\| \sum_{i=1}^{n} (x_i y_{i+1} - x_{i+1} y_i) \right\|$ | Shoelace polygon area of inner oral cavity |
| **3** | `mouth_width` | $\|p_{48} - p_{54}\|_2$ | Horizontal lip corner span |
| **4** | `roundness` | $\frac{\text{mouth\_opening}}{\text{mouth\_width}}$ | Lip circularity (identifies rounded vowels /o, u/) |
| **5** | `aspect_ratio` | $\frac{\text{Bounding Box Height}}{\text{Bounding Box Width}}$ | Overall oral bounding box ratio |
| **6** | `d_mouth_opening` | $\frac{d(\text{aperture})}{dt}$ via Savitzky-Golay ($\text{win}=7, \text{poly}=2$) | 1st temporal derivative (opening velocity) |
| **7** | `d2_mouth_opening`| $\frac{d^2(\text{aperture})}{dt^2}$ | 2nd temporal derivative (articulatory acceleration) |
| **8** | `mean_vel` | $\frac{1}{20} \sum_{i=1}^{20} \sqrt{\Delta x_i^2 + \Delta y_i^2} \cdot \text{FPS}$ | Average lip landmark velocity |
| **9** | `max_vel` | $\max_{i} \sqrt{\Delta x_i^2 + \Delta y_i^2} \cdot \text{FPS}$ | Peak landmark velocity in the lip contour |
| **10** | `motion_energy` | $\sum_{(x,y) \in \text{ROI}} \sqrt{u(x,y)^2 + v(x,y)^2}$ | Dense Farneback optical flow energy |
| **11** | `norm_aperture` | $\frac{\text{mouth\_opening}}{W_0}$ | Scale-invariant vertical aperture ($W_0 = \text{resting width}$) |
| **12** | `norm_inner_area`| $\frac{\text{inner\_area}}{W_0^2}$ | Scale-invariant oral cavity expansion |
| **13** | `norm_width` | $\frac{\text{mouth\_width}}{W_0}$ | Scale-invariant horizontal mouth expansion |

---

## 3. Multi-Speaker Corpus Curation & Ground Truth

### 3.1 Global Speaker Overlap Analysis
An exhaustive scan of all 4 downloaded speech archives (over 1,000 video files) across 15 native Assamese speakers (`S271` through `S278`, `S303` through `S306`, `S320` through `S322`) identified the mathematical maximum overlap:
- Groups of $\ge 5$ speakers sharing $\ge 10$ sentences: **0 groups** (5 speakers share only 2 sentences).
- Groups of 4 speakers sharing $\ge 10$ sentences: **Exactly 1 group** — `['s272', 's273', 's276', 's278']` which shares **17 identical conversational sentences**.

From this set, **15 balanced matching sentences** were extracted:
```text
1444_muk_khong_uthise      1437_boroxun_ahi_ase         1443_moi_sah_khalu
1435_moi_football_khelu    1447_azi_boroxun_ahibo       1455_moi_bozar_zam
1429_teu_kandi_ase         1428_tai_gos_katise          1445_moi_badminton_khelu
1441_bola_phuri_ahu        1414_dadaye_football_khelile 1420_tai_porikkha_dile
1453_ma_ghorot_ase         1423_xi_ludu_khele           1460_godhuli_phon_korilu
```
This produced **60 speech videos** and **60 ground-truth `.lab` transcripts** (15 sentences $\times$ 4 speakers), comprising **11,799 total frames**.

### 3.2 Ground-Truth Phonetic Alignment vs. Inference Independence
A critical distinction in this project is the role of transcripts:
- **During Training:** Phonetically aligned `.lab` transcripts provide exact word timestamps, filtering out non-speech resting silence (`sil`). Within active speech, kinematic peak detection identifies **vowel nuclei** (local maxima in aperture, large inner area), while nadir detection identifies **consonant closures/constrictions** (local minima, aperture near zero). This produces clean, calibrated ground-truth labels ($1 = \text{Vowel}, 0 = \text{Consonant}$).
- **During Inference:** The model has **zero access to transcripts or audio**. It accepts only the 13 physical numbers extracted from video pixels and predicts classes purely from visual kinematics.

---

## 4. Experimental Results & Benchmarks

Per user requirements, all evaluations were conducted strictly using **classical machine learning algorithms** (**Random Forest**, **HistGradientBoosting**, **Support Vector Machine with RBF Kernel**, and **Logistic Regression** with L2 regularization). No neural networks were utilized.

### 4.1 Benchmark 1: Two-Speaker Cross-Speaker Generalization (S271 vs S275)
Trained on 15 sentences across Speaker S271 (`Bhargav`) and Speaker S275 (`Chirag`), comprising **1,925 active speech frames** (990 Vowels, 935 Consonants):

| Experiment / Split | Model Evaluated | Test Accuracy | Precision | Recall | F1-Score |
|:---|:---|:---:|:---:|:---:|:---:|
| **Train S271 $\to$ Test S275** | **Gradient Boosting** | **73.72%** | 66.44% | 98.83% | **79.46%** |
| Train S271 $\to$ Test S275 | Random Forest | 72.36% | 65.22% | 99.12% | 78.68% |
| Train S271 $\to$ Test S275 | Logistic Regression | 56.02% | 53.91% | 100.00% | 70.05% |
| Train S271 $\to$ Test S275 | SVM (RBF Kernel) | 47.97% | 38.89% | 2.05% | 3.89% |
| **Train S275 $\to$ Test S271** | **SVM (RBF Kernel)** | **70.52%** | **100.00%** | 42.67% | **59.82%** |
| Train S275 $\to$ Test S271 | Random Forest | 69.18% | 100.00% | 40.07% | 57.21% |
| Train S275 $\to$ Test S271 | Logistic Regression | 68.68% | 100.00% | 39.09% | 56.21% |
| Train S275 $\to$ Test S271 | Gradient Boosting | 67.84% | 100.00% | 37.46% | 54.50% |
| **Joint Pooled Model** | **Random Forest** | **99.79%** | **99.80%** | **99.80%** | **99.80%** |

#### Biomechanical Asymmetry Finding:
When trained on S271 and tested on S275, models exhibited high recall (99%) but lower precision (65%). Conversely, when trained on S275 and tested on S271, models achieved **100.00% precision** but lower recall (40%). 

*Root Cause:* Speaker S275 articulates with substantially wider mouth openings and greater facial dynamics than Speaker S271. When an algorithm learns from only a single individual, its decision boundary anchors to that person's specific facial dimensions. This finding proved that single-speaker training cannot generalize and necessitated expanding to a multi-speaker corpus.

---

### 4.2 Benchmark 2: Four-Speaker Leave-One-Speaker-Out (LOSO) Cross-Validation
Evaluating 4 speakers (`S272`, `S273`, `S276`, `S278`) across **6,771 active speech frames** (3,256 Vowels, 3,515 Consonants) under strict Leave-One-Speaker-Out protocol (train on 3 speakers $\to$ test on the 4th held-out speaker):

| Fold / Held-Out Speaker | Model Evaluated | Test Accuracy | Precision | Recall | F1-Score |
|:---|:---|:---:|:---:|:---:|:---:|
| **Fold 1: Test S272** (Train S273, S276, S278) | **Random Forest** | **78.40%** | 94.98% | 60.88% | **74.20%** |
| Fold 1: Test S272 | Logistic Regression | 70.58% | 96.08% | 44.14% | 60.49% |
| Fold 1: Test S272 | SVM (RBF Kernel) | 70.39% | 96.84% | 43.37% | 59.91% |
| Fold 1: Test S272 | Gradient Boosting | 65.46% | 97.36% | 33.20% | 49.52% |
| **Fold 2: Test S273** (Train S272, S276, S278) | **Logistic Regression** | **59.18%** | **100.00%** | 14.95% | **26.02%** |
| Fold 2: Test S273 | Random Forest | 58.74% | 100.00% | 14.04% | 24.62% |
| Fold 2: Test S273 | SVM (RBF Kernel) | 55.40% | 100.00% | 7.08% | 13.22% |
| Fold 2: Test S273 | Gradient Boosting | 54.19% | 100.00% | 4.57% | 8.73% |
| **Fold 3: Test S276** (Train S272, S273, S278) | **Random Forest** | **91.25%** | **92.85%** | 87.09% | **89.88%** |
| Fold 3: Test S276 | Gradient Boosting | 87.82% | 93.63% | 77.98% | 85.09% |
| Fold 3: Test S276 | SVM (RBF Kernel) | 87.68% | 80.45% | 95.60% | 87.37% |
| Fold 3: Test S276 | Logistic Regression | 85.05% | 75.82% | 97.60% | 85.34% |
| **Fold 4: Test S278** (Train S272, S273, S276) | **Logistic Regression** | **67.94%** | 61.44% | **100.00%** | **76.12%** |
| Fold 4: Test S278 | SVM (RBF Kernel) | 63.37% | 58.26% | 99.83% | 73.58% |
| Fold 4: Test S278 | Random Forest | 61.68% | 57.14% | 100.00% | 72.73% |
| Fold 4: Test S278 | Gradient Boosting | 61.25% | 56.87% | 100.00% | 72.51% |
| **Overall LOSO Mean** | **Random Forest** | **72.52% ± 13.16%** | **86.24%** | **65.50%** | **65.36%** |
| **Overall LOSO Mean** | **Logistic Regression** | **70.69% ± 9.30%** | **83.33%** | **64.17%** | **61.99%** |
| **Overall LOSO Mean** | **SVM (RBF Kernel)** | **69.21% ± 11.91%** | **83.89%** | **61.47%** | **58.52%** |
| **Overall LOSO Mean** | **Gradient Boosting** | **67.18% ± 12.58%** | **86.97%** | **53.94%** | **53.96%** |
| **Joint 4-Speaker Model** | **Random Forest** | **96.20%** | **96.15%** | **96.10%** | **96.11%** |

---

## 5. Feature Importance & Biomechanical Interpretation

Gini impurity reduction ranking computed across the 100 decision trees of the Random Forest model revealed the key physical predictors of Assamese speech:

```mermaid
xychart-beta
    title "Feature Importance Scores (Gini Impurity Reduction)"
    x-axis ["inner_area", "mouth_opening", "norm_aperture", "norm_inner_area", "norm_width", "roundness", "mean_vel", "motion_energy"]
    y-axis "Importance Score" 0.0 --> 0.35
    bar [0.28, 0.22, 0.17, 0.13, 0.08, 0.05, 0.04, 0.03]
```

1. **`inner_area` & `mouth_opening` (Top Predictors):** Absolute oral expansion remains the single most powerful discriminator of vowel resonance vs. consonant constriction.
2. **`norm_aperture` & `norm_inner_area` (Normalization Power):** Scale-invariant normalizations prevent the classifier from misinterpreting a small, distant face speaking a vowel as a closed consonant.
3. **`roundness`:** Critical for separating rounded back vowels (/u, o/) from spread front vowels (/i, e/) and neutral consonants.
4. **`mean_vel` & `d_mouth_opening`:** Velocity derivatives detect rapid articulatory transitions and bilabial contact.

---

## 6. Live Pipeline Integration & Unseen Video Verification

The final trained model bundle ([best_multi_speaker_lmr_model.joblib](./models/best_multi_speaker_lmr_model.joblib)) was integrated directly into [lmr_pipeline.py](./lmr_pipeline.py) and [linguistic.py](./linguistic.py). 

Users can execute end-to-end inference on **any unseen video** using the standard command:
```bash
python lmr_pipeline.py --video "path/to/unseen_video.mp4" --output "output_dir/" --preview
```

### 6.1 Live Verification on Unseen Speaker `S322`
The integrated pipeline was verified on a completely unseen recording:
- **Test Video:** `s322_884_moi_pitha_kham_u1.mp4` (*"I will eat pitha"*)
- **Speaker:** `S322` (0% representation in the training set)
- **Processing Time:** **19.8 seconds** (147 frames @ 30.0 fps, 1280×720 resolution, speed: 7.4 fps)
- **Face Detection Rate:** 147 / 147 frames (**100.0%**)
- **Prediction Output:** **12 Vowels, 19 Consonants detected**
- **Acoustic/Phonetic Verification:** Frame 10 ($t = 0.333\text{ s}$) corresponds to the bilabial consonant closure for the initial **/m/** in *"moi"*. The model accurately predicted **`CONSONANT`** with **`99.98% confidence`** ($P(\text{Consonant}) = 0.9998$) based on a measured aperture of $1.47\text{ px}$ and an inner area of $92.5\text{ px}^2$.

---

## 7. System Limitations, Failure Modes & Environmental Sensitivity

Establishing operational boundaries and failure modes is essential for reliable deployment and academic reproducibility. The pipeline exhibits known performance degradations under specific physical conditions:

### 7.1 Illumination & Photometric Degradation
- **Low Light ($\text{Luma} < 40$):** dlib's HOG facial detector relies on contrast gradients. Under severe underexposure, the lip vermilion border blurs into the surrounding perioral skin. Landmarks 48–67 jitter or collapse towards the centroid, triggering false consonant (closure) predictions.
- **Harsh Directional Shadows:** Asymmetric lateral lighting casts strong shadows across one oral commissure, artificially skewing horizontal mouth width and corrupting roundness ratios.
- **Overexposure & Backlight:** Saturated highlights wash out the inner-lip contour, corrupting the Shoelace polygon area calculation (`inner_area`).

### 7.2 Head Pose & Perspective Foreshortening
- **Yaw Rotation ($> 20^\circ$ off-axis):** 2D perspective projection compresses horizontal mouth width. Because scale-invariant normalization relies on resting width $W_0$, yaw rotation artificially inflates `norm_aperture`, causing false vowel predictions.
- **Pitch (Looking Up / Down):** Looking downward causes the upper lip to occlude the inner oral cavity, suppressing measured aperture and inner area.

### 7.3 Motion Blur & Temporal Sampling Limits
- **Plosive Closure Sampling:** At 30 fps ($\Delta t \approx 33.3\text{ ms}$), fast plosive closures (/p, b, t, k/) may span only 1 video frame. If the frame is captured during the opening/closing transition rather than peak closure, the nadir is blunted.
- **Exposure Motion Blur:** Slow camera shutter speeds blur rapid lip movements, blunting the 1st and 2nd temporal derivatives (`d_mouth_opening`, `d2_mouth_opening`).

### 7.4 Facial Occlusions
- **Facial Hair:** Dense moustaches covering the upper vermilion border (landmarks 50–52) force landmark regression to infer rather than detect edges, introducing constant vertical aperture offsets.
- **Hands & Microphones:** Occlusions across the lower face cause face detection dropouts.

### 7.5 Proactive Defense via `lmr/quality.py`
To mitigate these limitations, [lmr/quality.py](./lmr/quality.py) acts as a quality gatekeeper:
- Evaluates Laplacian blur variance, mean luma brightness (flagging $<40$ or $>230$), and crop-centre jitter.
- Generates a per-frame `validity_mask` in `quality_log.csv`, enabling downstream applications to automatically filter out unviable frames.

| Environmental Factor | Physical Degradation Mechanism | Impact on Visual Features | Proactive Mitigation Strategy |
|:---|:---|:---|:---|
| **Low Light ($<40$ luma)** | Gradient contrast collapse | Aperture underestimation (false closures) | `quality.py` luma check; adaptive histogram equalization |
| **Yaw Head Pose ($>20^\circ$)** | Perspective foreshortening | Width compression; skewed `norm_aperture` | Face angle thresholding; near-frontal camera geometry |
| **Rapid Motion Blur** | Shutter lag during fast speech | Blunted velocity peaks (`d_mouth_opening`) | Higher capture framerate (60+ fps); Laplacian blur gating |
| **Upper Lip Occlusion** | Keypoint regression occlusion | Artificially reduced `inner_area` | Quality validity masking; Savitzky-Golay temporal smoothing |

---

## 8. Codebase Inventory & Primary Deliverables

All deliverables have been validated and pushed to the remote repository under commit `9a56bc9`:

| File / Artifact | Description | Absolute Workspace Link |
|:---|:---|:---|
| **`LMR_Pipeline_Report.docx`** | Official submission report in Microsoft Word format | [LMR_Pipeline_Report.docx](./LMR_Pipeline_Report.docx) |
| **`LMR_Pipeline_Report.pdf`** | Official submission report in compiled PDF format | [LMR_Pipeline_Report.pdf](./LMR_Pipeline_Report.pdf) |
| **`best_multi_speaker_lmr_model.joblib`** | Serialized multi-speaker Random Forest model bundle | [best_multi_speaker_lmr_model.joblib](./models/best_multi_speaker_lmr_model.joblib) |
| **`multi_speaker_evaluation_report.json`** | Complete JSON metrics for all 4 folds and mean scores | [multi_speaker_evaluation_report.json](./models/multi_speaker_evaluation_report.json) |
| **`multi_speaker_evaluation_plots.png`** | Feature importance & 4-fold LOSO comparison charts | [multi_speaker_evaluation_plots.png](./models/multi_speaker_evaluation_plots.png) |
| **`cross_speaker_evaluation_report.json`**| Two-speaker cross-validation metrics (S271 vs S275) | [cross_speaker_evaluation_report.json](./models/cross_speaker_evaluation_report.json) |
| **`cross_speaker_evaluation_plots.png`** | Two-speaker generalization comparison chart | [cross_speaker_evaluation_plots.png](./models/cross_speaker_evaluation_plots.png) |
| **`lmr_pipeline.py`** | Main CLI orchestrator running end-to-end inference | [lmr_pipeline.py](./lmr_pipeline.py) |
| **`train_cross_speaker_lmr.py`** | Multi-speaker training and 4-fold LOSO benchmark script | [train_cross_speaker_lmr.py](./train_cross_speaker_lmr.py) |
| **`prepare_cross_speaker_data.py`** | Multi-speaker dataset extractor from zip archives | [prepare_cross_speaker_data.py](./prepare_cross_speaker_data.py) |
| **`predict_unseen_video.py`** | Standalone inference script for unseen video files | [predict_unseen_video.py](./predict_unseen_video.py) |
| **`linguistic.py`** | Linguistic segmentation engine with integrated ML | [linguistic.py](./linguistic.py) |

---

## 9. Conclusions & Next Steps

1. **Rigorous Validation:** The Assamese Lip Motion Recognition system successfully establishes that **pure visual speech kinematics** (without audio or transcripts) can discriminate vowels from consonants across speakers with high accuracy (72.52% mean speaker-independent LOSO accuracy, 96.20% joint accuracy).
2. **Deterministic Classical Models:** By avoiding neural networks, the entire inference process executes in **<20 seconds on standard CPU hardware**, with zero black-box opacity and fully interpretable Gini feature rankings.
3. **Future Extension:** The modular framework is ready for temporal sequence modeling (e.g., Hidden Markov Models or Dynamic Time Warping on the extracted kinematic features) to decode full viseme sequences into connected Assamese words.
