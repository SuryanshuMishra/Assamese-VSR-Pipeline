"""
Lip Motion Recognition (LMR) Pipeline — Main Orchestrator
==========================================================
Wires together all vlr/ modules into a complete feature extraction pipeline.

Modules used:
    vlr.extraction      — FFmpeg frame extraction
    vlr.detection        — dlib landmark detection
    vlr.normalization    — Similarity-transform normalisation
    vlr.motion           — Per-keypoint motion features
    vlr.shape            — Mouth shape measurements
    vlr.optical_flow     — Dense optical flow + frame differencing
    vlr.quality          — Quality metrics + validity mask
    vlr.visualization    — Preview video + static plots

Usage:
    python lmr_pipeline.py --video <path> [--output <dir>] [--crop-size 96]
                           [--preview] [--debug]
"""

import sys
import json
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from vlr.extraction import get_video_info, frame_generator
from vlr.detection import LandmarkDetector
from vlr.normalization import normalize_frame
from vlr.motion import compute_motion_features, compute_total_distance
from vlr.shape import compute_shape_features
from vlr.optical_flow import compute_optical_flow_features, compute_flow_visualization
from vlr.quality import compute_quality_metrics, build_quality_log
from vlr.visualization import (
    draw_landmarks_on_frame,
    draw_displacement_vectors,
    create_preview_frame,
    plot_mouth_opening,
    plot_motion_energy,
    plot_lip_motion_magnitude,
    plot_trajectories,
    plot_displacement_heatmap,
    plot_raw_vs_normalized,
)


def process_video(video_path, output_dir, crop_size=96,
                  generate_preview=False, save_debug=False):
    """
    Full LMR pipeline:
      1. FFmpeg frame extraction (piped into memory)
      2. dlib landmark detection (confidence + failure tracking)
      3. Similarity-transform normalisation (translation + scale + rotation)
      4. Motion features (displacement, velocity, acceleration, trajectories)
      5. Shape features (opening, area, roundness, symmetry, derivatives)
      6. Optical flow + frame differencing on normalised mouth images
      7. Quality metrics + validity mask
      8. Visualisation (preview video + static plots)

    Exports:
      csvs/ (landmarks.csv, motion_features.csv, shape_features.csv,
             optical_flow_features.csv, quality_log.csv),
      graphs/ (mouth_opening.png, motion_energy.png, …),
      mouth_crops/, metadata.json, preview.mp4
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    # --- Output directories ---
    csvs_dir  = output_dir / "csvs"
    graphs_dir = output_dir / "graphs"
    crops_dir = output_dir / "mouth_crops"
    csvs_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    if save_debug:
        debug_dir = output_dir / "debug_frames"
        debug_dir.mkdir(parents=True, exist_ok=True)

    # === Stage 1: Video metadata ===
    info = get_video_info(video_path)
    fps    = info["fps"]
    width  = info["width"]
    height = info["height"]
    total_expected = info["total_frames"]

    if width == 0 or height == 0:
        print("[ERROR] Could not determine video resolution.")
        sys.exit(1)

    print(f"[INFO] Video: {video_path.name}")
    print(f"[INFO] Resolution: {width}x{height}, FPS: {fps:.2f}, "
          f"Expected frames: {total_expected or 'unknown'}")

    # === Stage 2: Load models ===
    det = LandmarkDetector()
    print("[INFO] dlib models loaded.")

    # === Stage 3: Process frames ===
    print("[INFO] Processing frames...")
    start_time = time.time()

    # Accumulators
    landmark_records   = []
    quality_records    = []
    norm_mouth_seq     = []   # normalised mouth landmarks per frame
    raw_mouth_seq      = []   # raw mouth landmarks per frame
    mouth_images       = []   # normalised mouth ROI images
    timestamps         = []   # actual timestamps for detected frames
    frames_processed   = 0
    frames_detected    = 0
    failed_indices     = []

    # Preview video writer
    preview_writer = None
    prev_norm_mouth = None
    prev_mouth_img  = None

    for frame_data in frame_generator(video_path, width, height, fps):
        idx = frame_data.index

        # --- Detection ---
        lm_result = det.detect(frame_data)

        # --- Normalisation ---
        norm_result = None
        if lm_result.detected:
            norm_result = normalize_frame(
                frame_data.bgr,
                lm_result.landmarks_68,
                lm_result.stable_refs,
                crop_size=crop_size,
            )

        # --- Quality ---
        q_row = compute_quality_metrics(frame_data.bgr, lm_result, norm_result)
        quality_records.append(q_row)

        # --- Collect data for detected frames ---
        if lm_result.detected and norm_result is not None:
            frames_detected += 1

            # Save normalised mouth crop
            crop_filename = f"crop_{idx:06d}.png"
            cv2.imwrite(str(crops_dir / crop_filename), norm_result.mouth_image)

            # Accumulate for batch feature computation
            norm_mouth_seq.append(norm_result.norm_mouth.copy())
            raw_mouth_seq.append(lm_result.mouth_landmarks.copy())
            mouth_images.append(norm_result.mouth_image.copy())
            timestamps.append(frame_data.timestamp_sec)

            # Build landmark record
            record = {
                "frame": idx,
                "timestamp_sec": frame_data.timestamp_sec,
                "detected": True,
                "confidence": lm_result.confidence,
                "face_angle_deg": round(norm_result.face_angle_deg, 4),
            }
            for i, (x, y) in enumerate(lm_result.mouth_landmarks):
                record[f"raw_x{i}"] = round(float(x), 4)
                record[f"raw_y{i}"] = round(float(y), 4)
            for i, (x, y) in enumerate(norm_result.norm_mouth):
                record[f"norm_x{i}"] = round(float(x), 4)
                record[f"norm_y{i}"] = round(float(y), 4)
            landmark_records.append(record)

            # --- Preview video (optional) ---
            if generate_preview:
                disp_vis = draw_displacement_vectors(
                    norm_result.mouth_image, prev_norm_mouth,
                    norm_result.norm_mouth
                )
                flow_vis = None
                if prev_mouth_img is not None:
                    flow_vis, _ = compute_flow_visualization(
                        prev_mouth_img, norm_result.mouth_image
                    )

                preview_frame = create_preview_frame(
                    frame_data.bgr, lm_result.landmarks_68,
                    norm_result.mouth_image, disp_vis, flow_vis,
                    idx, frame_data.timestamp_sec,
                )
                if preview_writer is None:
                    ph, pw = preview_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    preview_writer = cv2.VideoWriter(
                        str(output_dir / "preview.mp4"), fourcc, fps, (pw, ph)
                    )
                preview_writer.write(preview_frame)

                prev_norm_mouth = norm_result.norm_mouth.copy()
                prev_mouth_img  = norm_result.mouth_image.copy()

            # Debug frames
            if save_debug:
                vis = draw_landmarks_on_frame(frame_data.bgr, lm_result.landmarks_68)
                cv2.imwrite(str(debug_dir / f"debug_{idx:06d}.png"), vis)

        else:
            failed_indices.append(idx)
            landmark_records.append({
                "frame": idx,
                "timestamp_sec": frame_data.timestamp_sec,
                "detected": False,
                "confidence": lm_result.confidence,
                "failure_reason": lm_result.failure_reason,
            })

        frames_processed += 1

        # Progress
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            speed = (idx + 1) / elapsed if elapsed > 0 else 0
            total_disp = total_expected or "?"
            print(f"  [{idx+1}/{total_disp}] Detected: {frames_detected} | "
                  f"Speed: {speed:.1f} fps")

    if preview_writer is not None:
        preview_writer.release()

    elapsed = time.time() - start_time
    print(f"[INFO] Frame processing done in {elapsed:.1f}s "
          f"({frames_processed/elapsed:.1f} fps)")

    # === Stage 4: Batch feature computation ===
    print("[INFO] Computing motion features...")
    if len(norm_mouth_seq) > 1:
        motion_raw, motion_smoothed = compute_motion_features(
            norm_mouth_seq, timestamps
        )
        total_dist = compute_total_distance(norm_mouth_seq)
    else:
        motion_raw = motion_smoothed = pd.DataFrame()
        total_dist = np.zeros(20)

    print("[INFO] Computing shape features...")
    if len(norm_mouth_seq) > 0:
        shape_df = compute_shape_features(norm_mouth_seq, timestamps)
    else:
        shape_df = pd.DataFrame()

    print("[INFO] Computing optical flow features...")
    if len(mouth_images) > 1:
        flow_df = compute_optical_flow_features(mouth_images, timestamps)
    else:
        flow_df = pd.DataFrame()

    print("[INFO] Building quality log...")
    quality_df, quality_summary = build_quality_log(quality_records)

    # === Stage 5: Export CSVs ===
    print("[INFO] Saving outputs...")

    pd.DataFrame(landmark_records).to_csv(csvs_dir / "landmarks.csv", index=False)

    if not motion_raw.empty:
        motion_raw.to_csv(csvs_dir / "motion_features.csv", index=False)

    if not shape_df.empty:
        shape_df.to_csv(csvs_dir / "shape_features.csv", index=False)

    if not flow_df.empty:
        flow_df.to_csv(csvs_dir / "optical_flow_features.csv", index=False)

    quality_df.to_csv(csvs_dir / "quality_log.csv", index=False)

    # === Stage 6: Static plots ===
    print("[INFO] Generating plots...")

    if not shape_df.empty:
        plot_mouth_opening(shape_df, graphs_dir / "mouth_opening.png")

    if not flow_df.empty and "motion_energy" in flow_df.columns:
        plot_motion_energy(flow_df, graphs_dir / "motion_energy.png")

    if not motion_raw.empty and "mean_disp" in motion_raw.columns:
        plot_lip_motion_magnitude(motion_raw, graphs_dir / "lip_motion_magnitude.png")

    if len(norm_mouth_seq) > 1:
        plot_trajectories(norm_mouth_seq, graphs_dir / "trajectories.png")

    if not motion_raw.empty:
        plot_displacement_heatmap(motion_raw, graphs_dir / "displacement_heatmap.png")

    if len(raw_mouth_seq) > 1 and len(norm_mouth_seq) > 1:
        plot_raw_vs_normalized(raw_mouth_seq, norm_mouth_seq,
                               graphs_dir / "raw_vs_normalized.png")

    # === Stage 7: Metadata JSON ===
    metadata = {
        "video_file": str(video_path),
        "video_resolution": f"{width}x{height}",
        "video_fps": fps,
        "total_frames": frames_processed,
        "frames_detected": frames_detected,
        "frames_failed": len(failed_indices),
        "detection_rate": quality_summary.get("detection_rate", "0%"),
        "validity_rate": quality_summary.get("validity_rate", "0%"),
        "crop_size": crop_size,
        "pipeline": {
            "frame_extractor": "FFmpeg (piped to memory, no disk I/O)",
            "landmark_detector": "dlib 68-point frontal face predictor",
            "normalization": {
                "method": "Centroid mean-centring (translation removal)",
                "reference_points": "Mouth keypoints [48-67]",
                "preserves_shape": True,
                "description": ("Subtracts mouth centroid to centre lips at origin; "
                                "removes head translation while preserving genuine "
                                "lip deformation."),
            },
            "motion_features": [
                "displacement (dx, dy)", "magnitude", "direction",
                "velocity", "acceleration", "cumulative trajectory",
                "total distance", "mean/max displacement", "motion centroid",
            ],
            "shape_features": [
                "mouth width", "height", "opening", "aspect ratio",
                "inner area (Shoelace)", "perimeter", "roundness",
                "symmetry", "Δ(feature)", "Δ²(feature)",
            ],
            "optical_flow": [
                "Farneback dense flow", "frame differencing (MAD)",
                "flow magnitude", "motion energy",
                "horizontal/vertical energy",
            ],
            "quality_metrics": [
                "blur (Laplacian var)", "brightness", "face confidence",
                "validity mask", "failure reasons",
                "crop-centre jitter",
            ],
        },
        "quality_summary": quality_summary,
        "total_distance_per_keypoint": {
            f"kp_{i}": round(float(total_dist[i]), 4) for i in range(20)
        },
        "processing_time_sec": round(elapsed, 2),
        "failed_frame_indices": failed_indices[:50],
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # === Summary ===
    print(f"\n{'='*60}")
    print(f"  LMR Pipeline Complete — Full Feature Extraction")
    print(f"{'='*60}")
    print(f"  Total frames        : {frames_processed}")
    print(f"  Faces detected      : {frames_detected} "
          f"({quality_summary.get('detection_rate', '0%')})")
    print(f"  Valid frames        : {quality_summary.get('valid_frames', 0)} "
          f"({quality_summary.get('validity_rate', '0%')})")
    print(f"  Failed frames       : {len(failed_indices)}")
    print(f"  Processing time     : {elapsed:.1f}s "
          f"({frames_processed/elapsed:.1f} fps)")
    print(f"  Outputs:")
    print(f"    CSVs                : {csvs_dir}")
    print(f"    Graphs              : {graphs_dir}")
    print(f"    Mouth crops (96×96) : {crops_dir}")
    print(f"    Metadata JSON       : {output_dir / 'metadata.json'}")
    if generate_preview:
        print(f"    Preview video       : {output_dir / 'preview.mp4'}")
    print(f"{'='*60}\n")


# ============================================================================
# CLI Entry Point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=("Lip Motion Recognition (LMR) Pipeline: Full feature extraction — "
                     "landmarks, normalisation, motion, shape, optical flow, quality, "
                     "and visualisation.")
    )
    parser.add_argument(
        "--video", "-v", type=str,
        default=r"C:\Users\surya\Downloads\s320_1351_moi_doi_khalu_u1.mp4",
        help="Path to input video file",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="lmr_output",
        help="Output directory (default: lmr_output)",
    )
    parser.add_argument(
        "--crop-size", type=int, default=96,
        help="Mouth crop image size in pixels (default: 96)",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Generate 4-panel preview video",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Save annotated debug frames with landmarks drawn",
    )

    args = parser.parse_args()
    process_video(
        video_path=args.video,
        output_dir=args.output,
        crop_size=args.crop_size,
        generate_preview=args.preview,
        save_debug=args.debug,
    )


if __name__ == "__main__":
    main()
