"""
Frame Deduplication
===================
Compares consecutive extracted frames and drops near-identical ones.
Uses pixel-wise Mean Absolute Difference (MAD) on the mouth crop region
to detect frames with no meaningful lip movement change.

Usage:
    python deduplicate_frames.py [--threshold 2.0] [--use-crops]
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def compute_frame_difference(img1, img2):
    """
    Compute Mean Absolute Difference between two grayscale images.
    Returns a scalar — higher = more different.
    """
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    return diff.mean()


def deduplicate_frames(input_dir, output_dir, threshold=2.0, use_crops=False):
    """
    Compare consecutive frames and keep only those with meaningful change.

    Parameters:
        input_dir: directory with source frames (extracted_frames or mouth_crops)
        output_dir: directory to copy unique frames into
        threshold: MAD threshold — frames with diff < threshold are dropped
        use_crops: if True, compare mouth crops (more sensitive to lip changes)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather frame files
    patterns = ["frame_*.png", "crop_*.png"] if use_crops else ["frame_*.png"]
    files = []
    for pat in patterns:
        files.extend(sorted(input_dir.glob(pat)))

    if not files:
        print(f"[ERROR] No frame files found in {input_dir}")
        return

    print(f"[INFO] Input frames : {len(files)}")
    print(f"[INFO] Threshold    : {threshold} (MAD)")
    print(f"[INFO] Source       : {input_dir}")
    print(f"[INFO] Destination  : {output_dir}")
    print()

    # Always keep the first frame
    kept = [files[0]]
    dropped = []
    prev_img = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)

    diffs = []

    for i in range(1, len(files)):
        curr_img = cv2.imread(str(files[i]), cv2.IMREAD_GRAYSCALE)
        if curr_img is None:
            dropped.append(files[i])
            continue

        diff = compute_frame_difference(prev_img, curr_img)
        diffs.append(diff)

        if diff >= threshold:
            kept.append(files[i])
            prev_img = curr_img  # Update reference only when we keep a frame
        else:
            dropped.append(files[i])

    # Copy kept frames to output with sequential naming
    for idx, frame_path in enumerate(kept):
        dst = output_dir / f"unique_{idx:06d}.png"
        shutil.copy2(str(frame_path), str(dst))

    # Stats
    diffs = np.array(diffs)
    print(f"{'='*50}")
    print(f"  Deduplication Results")
    print(f"{'='*50}")
    print(f"  Original frames  : {len(files)}")
    print(f"  Kept (unique)    : {len(kept)} ({len(kept)/len(files)*100:.1f}%)")
    print(f"  Dropped (dupes)  : {len(dropped)} ({len(dropped)/len(files)*100:.1f}%)")
    print(f"  Diff stats       : min={diffs.min():.2f}, max={diffs.max():.2f}, "
          f"mean={diffs.mean():.2f}, median={np.median(diffs):.2f}")
    print(f"  Output           : {output_dir}")
    print(f"{'='*50}")

    return kept, dropped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drop near-identical consecutive frames.")
    parser.add_argument("--threshold", "-t", type=float, default=2.0,
                        help="MAD threshold below which frames are dropped (default: 2.0)")
    parser.add_argument("--use-crops", action="store_true",
                        help="Compare mouth crops instead of full frames (more sensitive to lip changes)")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Input directory (default: auto-detect from lmr_output)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory for unique frames")

    args = parser.parse_args()

    if args.use_crops:
        input_dir = args.input or r"q:\IITg intern\lmr_output\mouth_crops"
        output_dir = args.output or r"q:\IITg intern\lmr_output\unique_crops"
    else:
        input_dir = args.input or r"q:\IITg intern\lmr_output\extracted_frames"
        output_dir = args.output or r"q:\IITg intern\lmr_output\unique_frames"

    deduplicate_frames(input_dir, output_dir, threshold=args.threshold, use_crops=args.use_crops)
