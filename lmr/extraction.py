"""
Module 1: Frame Extraction
===========================
FFmpeg-based video frame extraction. Frames are piped directly into memory
as numpy arrays — no intermediate files written to disk.

Provides:
    - get_video_info(path) → metadata dict
    - frame_generator(path, width, height, fps) → yields FrameData objects
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FrameData:
    """A single video frame with metadata."""
    index: int              # 0-based frame index
    timestamp_sec: float    # index / fps
    bgr: np.ndarray         # Full resolution BGR frame (H, W, 3)
    fps: float              # Source video FPS


def get_video_info(video_path):
    """
    Query video metadata (width, height, fps, total frames) using ffprobe.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
        "-of", "json",
        str(video_path)
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        stream = data.get("streams", [{}])[0]

        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))

        r_fps = stream.get("r_frame_rate", "30/1")
        if "/" in r_fps:
            num, den = r_fps.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 30.0
        else:
            fps = float(r_fps)

        nb = stream.get("nb_frames")
        total_frames = int(nb) if nb and nb.isdigit() else None

        return {"width": width, "height": height, "fps": fps,
                "total_frames": total_frames}
    except Exception as e:
        print(f"[WARN] ffprobe failed: {e}")
        return {"width": 0, "height": 0, "fps": 30.0, "total_frames": None}


def frame_generator(video_path, width, height, fps):
    """
    Yield FrameData objects by piping FFmpeg raw output into memory.
    No intermediate files are written to disk.

    Parameters:
        video_path: path to input video
        width, height: frame dimensions (from ffprobe)
        fps: frames per second
    """
    frame_size = width * height * 3  # BGR24

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-v", "error",
        "-"
    ]

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        bufsize=frame_size * 2
    )

    frame_idx = 0
    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) != frame_size:
                break
            bgr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            yield FrameData(
                index=frame_idx,
                timestamp_sec=round(frame_idx / fps, 6) if fps > 0 else 0.0,
                bgr=bgr.copy(),
                fps=fps,
            )
            frame_idx += 1
    finally:
        proc.stdout.close()
        proc.wait()
