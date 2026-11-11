# Visual Speech Recognition (VSR) Pipeline for Assamese Language

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Institution](https://img.shields.io/badge/Institution-IIT%20Guwahati-orange.svg)](https://www.iitg.ac.in/)

An end-to-end, multi-modal feature extraction and linguistic boundary analysis pipeline for **Assamese Visual Speech Recognition (VSR)** and **Audio-Visual Speech Recognition (AVSR)**. Designed to extract high-fidelity geometric, kinematic, and appearance-based lip features from video recordings for speech and viseme recognition.

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
├── vlr/                                # Core VSR Package
│   ├── __init__.py                     # Package declaration
│   ├── extraction.py                   # FFmpeg memory-piped frame decoding
│   ├── detection.py                    # dlib 68-point face & lip tracking
│   ├── normalization.py                # Centroid mean-centring & 96x96 ROI cropping
│   ├── motion.py                       # Lip displacement, velocity, acceleration
│   ├── shape.py                        # Geometric lip features & temporal derivatives
│   ├── optical_flow.py                 # Farneback dense optical flow & frame diff
│   ├── quality.py                      # Blur, brightness, confidence & validity mask
│   └── visualization.py                # 6 diagnostic plots + 4-panel preview video
├── vsr_pipeline.py                     # Single-video processing orchestrator
├── generate_linguistic_analysis.py     # Phonetic ground-truth (.lab) boundary alignment
├── deduplicate_frames.py               # Frame deduplication utility
├── VSR_Pipeline_Report.docx            # Full technical development report
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/SuryanshuMishra/Assamese-VSR-Pipeline.git
cd Assamese-VSR-Pipeline

pip install opencv-python dlib numpy pandas matplotlib scipy python-docx
```

> **Note**: `dlib` shape predictor model (`shape_predictor_68_face_landmarks.dat`) is automatically downloaded on the first run if missing.

### 2. Video Processing

Extract all kinematic, shape, optical flow, and visual diagnostic outputs from an input video:

```bash
python vsr_pipeline.py --video path/to/video.mp4 --output results/ --preview
```

### 3. Linguistic Boundary & Viseme Alignment

Correlate ground-truth speech transcripts (`.lab` files) with kinematic lip gestures, calculating aperture opening, roundness, and visual preparatory lead times:

```bash
python generate_linguistic_analysis.py
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

- **Language**: Assamese (অসমীয়া)
- **Institution**: Indian Institute of Technology Guwahati (IIT Guwahati)
- **Target Application**: Low-resource Lip Motion Recognition & Speech Recognition
