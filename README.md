# Visual Speech Recognition (VSR) Pipeline for Assamese Language

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Institution](https://img.shields.io/badge/Institution-IIT%20Guwahati-orange.svg)](https://www.iitg.ac.in/)

An end-to-end, multi-modal feature extraction pipeline for **Assamese Visual Speech Recognition (VSR)** and **Audio-Visual Speech Recognition (AVSR)**. Designed to process large-scale multi-speaker speech corpora (424 video recordings across 4 speakers and 2 utterance sets) directly from Google Drive or local storage.

---

## 🌟 Architecture Overview

```
                      Raw Video (Assamese Speech)
                                 │
                     [1. Frame Extraction] (FFmpeg)
                     Pipes raw BGR frames to memory
                                 │
                    [2. Landmark Detection] (dlib)
                     68 face landmarks + 20 lip points
                                 │
                      [3. Normalisation]
                     Mouth centroid mean-centring
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
  [4. Motion Features]   [5. Shape Features]    [6. Optical Flow]
  • Per-keypoint dx/dy   • Inner area (Shoelace)• Dense Farneback flow
  • Velocity & Accel     • Opening & Roundness  • Motion energy (HSV)
  • Trajectories         • Aspect ratio & Δ, Δ² • Frame differencing (MAD)
         └───────────────────────┼───────────────────────┘
                                 │
                     [7. Quality Gating]
                     Blur, brightness, confidence & validity mask
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
  [Structured Exports]                           [Visual Diagnostics]
  • landmarks.csv                                • mouth_opening.png
  • motion_features.csv                          • motion_energy.png
  • shape_features.csv                           • lip_motion_magnitude.png
  • optical_flow_features.csv                    • trajectories.png (6 keypoints)
  • quality_log.csv                              • displacement_heatmap.png
  • mouth_crops/ (96×96 PNGs)                    • raw_vs_normalized.png
  • metadata.json                                • preview.mp4 (4-panel video)
```

---

## 📁 Repository Structure

```
├── vlr/                        # Core VSR Package
│   ├── __init__.py             # Package declaration
│   ├── extraction.py           # FFmpeg memory-piped frame decoding
│   ├── detection.py            # dlib 68-point face & lip tracking
│   ├── normalization.py        # Centroid mean-centring & 96x96 ROI cropping
│   ├── motion.py               # Lip displacement, velocity, acceleration
│   ├── shape.py                # Geometric lip features & temporal derivatives
│   ├── optical_flow.py         # Farneback dense optical flow & frame diff
│   ├── quality.py              # Blur, brightness, confidence & validity mask
│   └── visualization.py        # 6 diagnostic plots + 4-panel preview video
├── vsr_pipeline.py             # Single-video processing orchestrator
├── drive_batch_pipeline.py     # Cloud-native Google Drive batch processor
├── deduplicate_frames.py       # Frame deduplication utility
├── upload_existing_results.py  # Direct Drive uploader for processed batches
├── VSR_Pipeline_Report.docx    # Full technical development report
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

pip install opencv-python dlib numpy pandas matplotlib scipy docx
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

> **Note**: `dlib` shape predictor model (`shape_predictor_68_face_landmarks.dat`) is automatically downloaded on the first run if missing.

### 2. Single Video Processing

```bash
python vsr_pipeline.py --video path/to/video.mp4 --output results/ --preview
```

### 3. Google Drive Batch Processing (Multi-Speaker Corpus)

Processes the Assamese corpus (`GCU/Suryanshu Mishra/` with speakers `S319F`, `S320`, `S321`, `S322` across `u1` and `u2`):

```bash
python drive_batch_pipeline.py \
    --credentials credentials.json \
    --root-path "GCU/Suryanshu Mishra" \
    --speakers S319F S320 S321 S322 \
    --utterances u1 u2 \
    --dest-path "GCU/Suryanshu Mishra/vsr_results"
```

---

## 📊 Features Extracted

| Modality | Features | Columns |
|---|---|---|
| **Landmarks** | Centered $(x,y)$ coordinates for 20 lip keypoints | 44 |
| **Motion** | Velocity, acceleration, displacement magnitude & angle | ~227 |
| **Shape** | Aperture opening, inner area (Shoelace), roundness, $\Delta$, $\Delta^2$ | 24 |
| **Optical Flow** | Farneback magnitude, horizontal/vertical energy, MAD frame differencing | 12 |
| **Quality** | Laplacian blur score, brightness, face confidence, validity flag | 12 |

---

## 🏛️ Project Details

- **Language**: Assamese
- **Institution**: Indian Institute of Technology Guwahati (IIT Guwahati)
- **Target Application**: Low-resource Visual Lip Reading & Speech Recognition
