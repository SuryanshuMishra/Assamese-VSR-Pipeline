"""
Google Drive ↔ VSR Pipeline — Batch Processor (OAuth 2.0)
==========================================================
Traverses a Google Drive folder tree like:

    GCU / Suryanshu Mishra / s319f / u1 / video / *.mp4
                              ^^^^   ^^
                              speaker utterance

Each utterance folder (u1, u2, …) contains three subfolders:
    audio/   — audio files
    video/   — video files (50 per utterance, named after the sentence)
    text/    — text transcripts

… downloads each video, runs the full VSR pipeline, and writes
all outputs into three GLOBAL folders on Drive (+ metadata):

    vsr_results/
    ├── landmarks/          ← one CSV per video
    ├── graphs/             ← one sub-folder per video (all plots)
    ├── mouth_crops/        ← one sub-folder per video (96×96 PNGs)
    └── metadata/           ← one JSON per video

File-naming scheme:
    s319_u1_moi_doi_khalu.csv
    s319_u1_moi_doi_khalu/  (for graphs & crops sub-folders)

Authentication
--------------
Uses OAuth 2.0 "Installed App" flow — your personal Google account.
On the first run a browser window opens; a refresh-token is cached
in ``token.json`` so subsequent runs are headless.

Setup (one-time)
~~~~~~~~~~~~~~~~
1. Go to https://console.cloud.google.com/
2. Create a project → enable the **Google Drive API**.
3. APIs & Services → Credentials → + Create Credentials
   → **OAuth client ID** → Application type = **Desktop app**.
4. Download the JSON → save as ``credentials.json`` next to this script.
5. Install dependencies::

       pip install google-api-python-client google-auth-httplib2 \\
                   google-auth-oauthlib

Usage
~~~~~
::

    python drive_batch_pipeline.py \\
        --credentials  credentials.json \\
        --root-path    "GCU/Suryanshu Mishra" \\
        --speakers     s319f s320f \\
        --utterances   u1 u2 u3 \\
        --dest-path    "GCU/Suryanshu Mishra/vsr_results" \\
        [--crop-size 96] [--preview] [--keep-local]
"""

from __future__ import annotations

import os
import re
import sys
import json
import shutil
import argparse
import tempfile
from pathlib import Path

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

# ── Import the existing pipeline ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vsr_pipeline import process_video


# =====================================================================
# Google Drive Helper (OAuth 2.0)
# =====================================================================
class DriveManager:
    """Google Drive API v3 wrapper — OAuth 2.0 Installed App flow."""

    SCOPES = ["https://www.googleapis.com/auth/drive"]

    VIDEO_MIMES = {
        "video/mp4", "video/x-matroska", "video/avi",
        "video/quicktime", "video/webm", "video/x-msvideo",
        "video/x-flv", "video/mpeg",
    }

    def __init__(self, credentials_path: str, token_path: str = "token.json"):
        self.credentials_path = credentials_path
        self.token_path = token_path
        creds = self._authenticate()
        self.service = build("drive", "v3", credentials=creds)

    # ── OAuth 2.0 authentication ──────────────────────────────────
    def _authenticate(self) -> Credentials:
        """Authenticate via OAuth 2.0 Installed App flow.
        Caches refresh-token in token.json for headless re-runs.
        """
        creds = None

        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(
                self.token_path, self.SCOPES
            )

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, self.SCOPES
                )
                creds = flow.run_local_server(port=0)

            with open(self.token_path, "w") as f:
                f.write(creds.to_json())

        return creds

    # ── Resolve a path like "GCU/Suryanshu Mishra/s319f" to a folder ID ──
    def resolve_path(self, path: str, create: bool = False) -> str | None:
        """Walk a slash-separated path from the Drive root and return
        the folder ID of the final component.

        Supports:
          - Shared folders (e.g. 'GCU' in 'Shared with me')
          - Case-insensitive and space-insensitive folder name matching (e.g. 'S319F' matches 'S319 F')
          - Direct Google Drive folder IDs
        """
        path_str = path.strip().replace("\\", "/")
        # If user passed a raw Google Drive Folder ID
        if "/" not in path_str and len(path_str) >= 25 and " " not in path_str:
            try:
                f = self.service.files().get(fileId=path_str, fields="id, trashed").execute()
                if not f.get("trashed"):
                    return f["id"]
            except Exception:
                pass

        parts = [p.strip() for p in path_str.split("/") if p.strip()]
        if not parts:
            return "root"

        current_id = "root"

        for i, part in enumerate(parts):
            subfolders = self.list_subfolders(current_id) if current_id != "root" else []
            
            # If current_id is root, also search across all drives / shared with me
            if current_id == "root":
                query = (
                    f"name = '{part}' and "
                    f"mimeType = 'application/vnd.google-apps.folder' and "
                    f"trashed = false"
                )
                resp = self.service.files().list(
                    q=query, fields="files(id, name, parents)", pageSize=10
                ).execute()
                files = resp.get("files", [])
                if files:
                    current_id = files[0]["id"]
                    continue
                elif create:
                    current_id = self._create_folder(part, current_id)
                    continue
                else:
                    return None

            # For subsequent components under current_id, do flexible matching
            norm_target = re.sub(r"\s+", "", part).lower()
            matched_id = None
            for sf in subfolders:
                norm_sf = re.sub(r"\s+", "", sf["name"]).lower()
                if norm_sf == norm_target:
                    matched_id = sf["id"]
                    break

            if matched_id:
                current_id = matched_id
            elif create:
                current_id = self._create_folder(part, current_id)
            else:
                return None

        return current_id

    def _create_folder(self, name: str, parent_id: str) -> str:
        meta = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        }
        folder = self.service.files().create(
            body=meta, fields="id"
        ).execute()
        return folder["id"]

    # ── List videos in a folder ───────────────────────────────────
    def list_videos(self, folder_id: str) -> list[dict]:
        results = []
        page_token = None

        while True:
            query = f"'{folder_id}' in parents and trashed = false"
            resp = self.service.files().list(
                q=query,
                fields="nextPageToken, files(id, name, mimeType, size)",
                pageSize=100,
                pageToken=page_token,
            ).execute()

            for f in resp.get("files", []):
                if f["mimeType"] in self.VIDEO_MIMES:
                    results.append(f)

            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        return results

    # ── Download ──────────────────────────────────────────────────
    def download_file(self, file_id: str, dest_path: str):
        request = self.service.files().get_media(fileId=file_id)
        with open(dest_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    pct = int(status.progress() * 100)
                    print(f"    ↓ {pct}%", end="\r")
        print()

    # ── Get or create folder ──────────────────────────────────────
    def get_or_create_folder(self, name: str, parent_id: str) -> str:
        query = (
            f"'{parent_id}' in parents and "
            f"name = '{name}' and "
            f"mimeType = 'application/vnd.google-apps.folder' and "
            f"trashed = false"
        )
        resp = self.service.files().list(
            q=query, fields="files(id)"
        ).execute()
        files = resp.get("files", [])
        if files:
            return files[0]["id"]
        return self._create_folder(name, parent_id)

    # ── Upload a single file ──────────────────────────────────────
    def upload_file(self, local_path: str, parent_id: str,
                    mime_type: str | None = None):
        name = os.path.basename(local_path)
        if mime_type is None:
            mime_type = self._guess_mime(name)

        meta = {"name": name, "parents": [parent_id]}
        media = MediaFileUpload(local_path, mimetype=mime_type, resumable=True)

        request = self.service.files().create(
            body=meta, media_body=media, fields="id, name"
        )
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                print(f"    ↑ {name} … {pct}%", end="\r")
        print(f"    ↑ Uploaded: {name}")

    # ── Upload a directory tree ───────────────────────────────────
    def upload_directory(self, local_dir: str, parent_id: str):
        local_dir = Path(local_dir)
        for item in sorted(local_dir.iterdir()):
            if item.is_dir():
                sub_id = self.get_or_create_folder(item.name, parent_id)
                self.upload_directory(str(item), sub_id)
            else:
                self.upload_file(str(item), parent_id)

    # ── List sub-folders in a folder ─────────────────────────────
    def list_subfolders(self, folder_id: str) -> list[dict]:
        """Return [{id, name}, …] for all sub-folders in *folder_id*."""
        results = []
        page_token = None

        while True:
            query = (
                f"'{folder_id}' in parents and "
                f"mimeType = 'application/vnd.google-apps.folder' and "
                f"trashed = false"
            )
            resp = self.service.files().list(
                q=query,
                fields="nextPageToken, files(id, name)",
                pageSize=100,
                pageToken=page_token,
            ).execute()

            results.extend(resp.get("files", []))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        return sorted(results, key=lambda f: f["name"])

    @staticmethod
    def _guess_mime(filename: str) -> str:
        ext = Path(filename).suffix.lower()
        mapping = {
            ".csv":  "text/csv",
            ".json": "application/json",
            ".png":  "image/png",
            ".jpg":  "image/jpeg",
            ".jpeg": "image/jpeg",
            ".mp4":  "video/mp4",
            ".avi":  "video/x-msvideo",
            ".mkv":  "video/x-matroska",
            ".mov":  "video/quicktime",
            ".webm": "video/webm",
        }
        return mapping.get(ext, "application/octet-stream")


# =====================================================================
# Output Re-organiser
# =====================================================================
def reorganize_outputs(pipeline_output_dir: Path,
                       global_output_dir: Path,
                       prefix: str):
    """
    Move outputs from the per-video pipeline output into a flat global
    structure:

        global_output_dir/
        ├── landmarks/  {prefix}.csv              (renamed from landmarks.csv)
        ├── graphs/     {prefix}/  *.png           (all plots)
        ├── mouth_crops/{prefix}/  crop_*.png      (all crops)
        └── metadata/   {prefix}.json              (metadata.json)

    Other CSVs (motion_features, shape_features, optical_flow_features,
    quality_log) are merged into a single CSV or stored alongside
    landmarks.
    """
    landmarks_dir  = global_output_dir / "landmarks"
    graphs_dir     = global_output_dir / "graphs" / prefix
    crops_dir      = global_output_dir / "mouth_crops" / prefix
    metadata_dir   = global_output_dir / "metadata"

    landmarks_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    src = pipeline_output_dir

    # --- CSVs → landmarks/ with prefix ---
    csvs_src = src / "csvs"
    if csvs_src.exists():
        for csv_file in csvs_src.glob("*.csv"):
            dest_name = f"{prefix}_{csv_file.name}"
            shutil.copy2(str(csv_file), str(landmarks_dir / dest_name))

    # --- Graphs → graphs/{prefix}/ ---
    graphs_src = src / "graphs"
    if graphs_src.exists():
        for plot_file in graphs_src.glob("*.png"):
            shutil.copy2(str(plot_file), str(graphs_dir / plot_file.name))

    # --- Mouth crops → mouth_crops/{prefix}/ ---
    crops_src = src / "mouth_crops"
    if crops_src.exists():
        for crop_file in crops_src.glob("*.png"):
            shutil.copy2(str(crop_file), str(crops_dir / crop_file.name))

    # --- Metadata → metadata/{prefix}.json ---
    meta_src = src / "metadata.json"
    if meta_src.exists():
        shutil.copy2(str(meta_src), str(metadata_dir / f"{prefix}.json"))

    # --- Preview video (if generated) ---
    preview_src = src / "preview.mp4"
    if preview_src.exists():
        previews_dir = global_output_dir / "previews"
        previews_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(preview_src),
                      str(previews_dir / f"{prefix}.mp4"))


# =====================================================================
# Helpers
# =====================================================================
def parse_speaker_utterance(speaker_folder: str, utterance_folder: str) -> str:
    """Build a clean prefix from folder names, removing whitespace.

    ``S319 F`` + ``u1`` → ``S319F_u1``
    """
    clean_spk = re.sub(r"\s+", "", speaker_folder)
    clean_utt = re.sub(r"\s+", "", utterance_folder)
    return f"{clean_spk}_{clean_utt}"


def sanitize_video_name(name: str) -> str:
    """Turn a video filename into a safe, filesystem-friendly string.

    ``s320_1351_moi doi khalu_u1.mp4``  →  ``s320_1351_moi_doi_khalu_u1``
    """
    stem = Path(name).stem
    # Replace spaces and problematic chars with underscores
    safe = stem.replace(" ", "_").replace("-", "_")
    # Collapse multiple underscores
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")


# =====================================================================
# Batch Orchestrator
# =====================================================================
def run_batch(
    credentials_path: str,
    root_path: str,
    speakers: list[str] | None,
    utterances: list[str] | None,
    dest_path: str,
    crop_size: int = 96,
    generate_preview: bool = False,
    keep_local: bool = False,
    local_work_dir: str | None = None,
):
    """
    End-to-end batch workflow:
        1. Authenticate with OAuth 2.0
        2. Auto-discover or use specified speaker/utterance folders
        3. For each speaker + utterance combo, find the video/ folder
        4. Download each video, run the pipeline
        5. Re-organize outputs into global folders
        6. Upload the global output tree to Drive
    """
    drive = DriveManager(credentials_path)

    # ── Resolve root folder on Drive ──────────────────────────────
    root_id = drive.resolve_path(root_path)
    if root_id is None:
        print(f"[ERROR] Root path not found on Drive: {root_path}")
        sys.exit(1)

    # ── Auto-discover speakers if not explicitly provided ─────────
    if speakers is None:
        print("[DRIVE] Auto-discovering speaker folders …")
        all_folders = drive.list_subfolders(root_id)
        # Match folders starting with 's' followed by digits,
        # optionally ending with 'f' (female) or 'm' (male)
        import re
        speaker_pattern = re.compile(r"^s\d+[a-zA-Z]?$", re.IGNORECASE)
        speakers = [f["name"] for f in all_folders
                    if speaker_pattern.match(f["name"])]
        if not speakers:
            print("[ERROR] No speaker folders (s###, s###f, …) found "
                  f"under: {root_path}")
            sys.exit(1)
        print(f"[DRIVE] Found {len(speakers)} speaker(s): {speakers}")

    # ── Auto-discover utterances if not explicitly provided ───────
    # We'll discover per-speaker if needed
    auto_discover_utterances = (utterances is None)
    if auto_discover_utterances:
        print("[DRIVE] Will auto-discover utterance folders per speaker.")

    # ── Working directory ─────────────────────────────────────────
    if local_work_dir:
        work_root = Path(local_work_dir)
        work_root.mkdir(parents=True, exist_ok=True)
        use_temp = False
    else:
        work_root = Path(tempfile.mkdtemp(prefix="vsr_drive_"))
        use_temp = True

    global_output = work_root / "vsr_results"
    global_output.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Local working directory: {work_root}")
    print(f"[INFO] Global output directory: {global_output}")

    # ── Discover & process ────────────────────────────────────────
    total_videos = 0
    success_count = 0
    fail_count = 0
    skipped_folders = []

    # ── Upload global results destination check ───────────────────
    dest_folder_id = drive.resolve_path(dest_path, create=True)
    meta_drive_folder_id = drive.get_or_create_folder("metadata", dest_folder_id)
    # Cache existing metadata files on Drive for instantaneous skip/resume
    existing_drive_meta = set()
    try:
        query = f"'{meta_drive_folder_id}' in parents and trashed = false"
        resp = drive.service.files().list(q=query, fields="files(name)", pageSize=1000).execute()
        existing_drive_meta = {f["name"] for f in resp.get("files", [])}
        if existing_drive_meta:
            print(f"[RESUME] Found {len(existing_drive_meta)} videos already uploaded on Drive. Will skip them automatically.")
    except Exception as e:
        print(f"[WARN] Could not fetch existing files on Drive: {e}")

    for speaker in speakers:
        # Resolve speaker folder
        speaker_id = drive.resolve_path(f"{root_path}/{speaker}")
        if speaker_id is None:
            print(f"\n  [SKIP] Speaker folder not found: {root_path}/{speaker}")
            skipped_folders.append(f"{root_path}/{speaker}")
            continue

        # Discover utterance folders if not explicitly given
        if auto_discover_utterances:
            import re
            all_utt = drive.list_subfolders(speaker_id)
            utt_pattern = re.compile(r"^u\d+$", re.IGNORECASE)
            current_utterances = [f["name"] for f in all_utt
                                  if utt_pattern.match(f["name"])]
            if not current_utterances:
                print(f"\n  [SKIP] No utterance folders (u1, u2, …) "
                      f"found under: {speaker}")
                skipped_folders.append(f"{root_path}/{speaker}")
                continue
            # Sort numerically: u1, u2, …, u10, u11
            current_utterances.sort(key=lambda x: int(x[1:]))
            print(f"\n  [{speaker}] Found utterances: {current_utterances}")
        else:
            current_utterances = utterances

        for utterance in current_utterances:
            # Build the Drive path: root_path / speaker / utterance / video
            folder_path = f"{root_path}/{speaker}/{utterance}/video"
            print(f"\n{'━'*60}")
            print(f"  Scanning: {folder_path}")
            print(f"{'━'*60}")

            folder_id = drive.resolve_path(folder_path)
            if folder_id is None:
                print(f"  [SKIP] Folder not found on Drive: {folder_path}")
                skipped_folders.append(folder_path)
                continue

            videos = drive.list_videos(folder_id)
            if not videos:
                print(f"  [SKIP] No videos found in: {folder_path}")
                skipped_folders.append(folder_path)
                continue

            print(f"  Found {len(videos)} video(s)")

            # Prefix for this speaker+utterance combo
            su_prefix = parse_speaker_utterance(speaker, utterance)

            for i, video_info in enumerate(videos, start=1):
                video_name = video_info["name"]
                video_stem = sanitize_video_name(video_name)
                video_id   = video_info["id"]

                # Full prefix: s319_u1_sentence_name
                full_prefix = f"{su_prefix}_{video_stem}"

                # ── Resume check: Skip if already done on Drive or locally ──
                expected_json = f"{full_prefix}.json"
                local_meta = global_output / "metadata" / expected_json
                if expected_json in existing_drive_meta or local_meta.exists():
                    print(f"  [SKIP] Video already completed: {full_prefix}")
                    total_videos += 1
                    success_count += 1
                    continue

                total_videos += 1
                print(f"\n  [{i}/{len(videos)}] {video_name}")
                print(f"  Prefix: {full_prefix}")

                download_path = work_root / "downloads" / video_name
                download_path.parent.mkdir(parents=True, exist_ok=True)
                pipeline_out  = work_root / "pipeline_tmp" / full_prefix

                try:
                    # ── Download ──────────────────────────────────
                    print(f"  [DRIVE] Downloading …")
                    drive.download_file(video_id, str(download_path))

                    # ── Run pipeline ──────────────────────────────
                    print(f"  [PIPELINE] Processing …")
                    process_video(
                        video_path=str(download_path),
                        output_dir=str(pipeline_out),
                        crop_size=crop_size,
                        generate_preview=generate_preview,
                        save_debug=False,
                    )

                    # ── Re-organize into global folders ───────────
                    print(f"  [ORGANIZE] Moving to global folders …")
                    reorganize_outputs(pipeline_out, global_output, full_prefix)

                    print(f"  [✓] {video_name} — done.")
                    success_count += 1

                except Exception as exc:
                    print(f"  [✗] {video_name} — FAILED: {exc}")
                    import traceback
                    traceback.print_exc()
                    fail_count += 1

                finally:
                    # Clean up per-video temp files
                    if not keep_local:
                        if download_path.exists():
                            download_path.unlink()
                        if pipeline_out.exists():
                            shutil.rmtree(pipeline_out)

    # ── Upload global results to Drive ────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Uploading results to Drive …")
    print(f"{'═'*60}")

    dest_folder_id = drive.resolve_path(dest_path, create=True)
    print(f"  Destination: {dest_path} (ID: {dest_folder_id})")

    # Upload each global sub-folder
    for sub in ["landmarks", "graphs", "mouth_crops", "metadata", "previews"]:
        sub_dir = global_output / sub
        if sub_dir.exists() and any(sub_dir.iterdir()):
            print(f"\n  Uploading {sub}/ …")
            sub_folder_id = drive.get_or_create_folder(sub, dest_folder_id)
            drive.upload_directory(str(sub_dir), sub_folder_id)

    # ── Cleanup ───────────────────────────────────────────────────
    if not keep_local:
        if use_temp:
            shutil.rmtree(work_root, ignore_errors=True)
        else:
            # Keep work_root but clean pipeline_tmp and downloads
            for d in ["pipeline_tmp", "downloads"]:
                p = work_root / d
                if p.exists():
                    shutil.rmtree(p, ignore_errors=True)

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Batch Complete")
    print(f"{'═'*60}")
    print(f"  Total videos processed : {total_videos}")
    print(f"  Succeeded              : {success_count}")
    print(f"  Failed                 : {fail_count}")
    if skipped_folders:
        print(f"  Skipped folders ({len(skipped_folders)}):")
        for sf in skipped_folders:
            print(f"    • {sf}")
    print(f"\n  Results uploaded to: {dest_path}")
    print(f"{'═'*60}\n")


# =====================================================================
# CLI
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Batch VSR pipeline: download videos from Google Drive, "
            "process them, and upload organized results back."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Auto-discover ALL speakers and utterances
  python drive_batch_pipeline.py \\
      --credentials credentials.json \\
      --root-path "GCU/Suryanshu Mishra" \\
      --dest-path "vsr_results"

  # Process specific speakers (mix of male & female)
  python drive_batch_pipeline.py \\
      --credentials credentials.json \\
      --root-path "GCU/Suryanshu Mishra" \\
      --speakers S319F S320 S321 S322 \\
      --utterances u1 u2 \\
      --dest-path "vsr_results"

  # Auto-discover speakers but process only u1 and u2
  python drive_batch_pipeline.py \\
      --credentials credentials.json \\
      --root-path "GCU/Suryanshu Mishra" \\
      --utterances u1 u2 \\
      --dest-path "GCU/Suryanshu Mishra/vsr_results" \\
      --preview --keep-local --work-dir ./local_vsr
        """,
    )

    parser.add_argument(
        "--credentials", "-c", required=True,
        help="Path to OAuth 2.0 client secrets JSON file (credentials.json)",
    )
    parser.add_argument(
        "--root-path", "-r", required=True,
        help='Drive path containing speaker folders (e.g. "GCU/Suryanshu Mishra")',
    )

    # Speaker selection — explicit list OR auto-discover (default)
    spk = parser.add_mutually_exclusive_group()
    spk.add_argument(
        "--speakers", nargs="+",
        help="Explicit speaker folder names (e.g. s319f s320 s321f). "
             "Omit to auto-discover all s### and s###f folders.",
    )

    # Utterance selection — explicit list OR auto-discover (default)
    utt = parser.add_mutually_exclusive_group()
    utt.add_argument(
        "--utterances", nargs="+",
        help="Explicit utterance folder names (e.g. u1 u2 u3). "
             "Omit to auto-discover all u# folders per speaker.",
    )

    parser.add_argument(
        "--dest-path", "-d", default="vsr_results",
        help='Drive path for results (default: "vsr_results" in your My Drive)',
    )
    parser.add_argument(
        "--crop-size", type=int, default=96,
        help="Mouth crop image size in pixels (default: 96)",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Generate 4-panel preview video per input",
    )
    parser.add_argument(
        "--keep-local", action="store_true",
        help="Keep local copies of all outputs",
    )
    parser.add_argument(
        "--work-dir", type=str, default=None,
        help="Local working directory (default: system temp dir)",
    )

    args = parser.parse_args()

    # Use explicit lists or None (auto-discover)
    speakers = args.speakers        # None triggers auto-discovery
    utterances = args.utterances    # None triggers auto-discovery

    print(f"[CONFIG] Speakers   : {speakers or 'auto-discover'}")
    print(f"[CONFIG] Utterances : {utterances or 'auto-discover'}")
    print(f"[CONFIG] Source root : {args.root_path}")
    print(f"[CONFIG] Destination: {args.dest_path}")

    run_batch(
        credentials_path=args.credentials,
        root_path=args.root_path,
        speakers=speakers,
        utterances=utterances,
        dest_path=args.dest_path,
        crop_size=args.crop_size,
        generate_preview=args.preview,
        keep_local=args.keep_local,
        local_work_dir=args.work_dir,
    )
