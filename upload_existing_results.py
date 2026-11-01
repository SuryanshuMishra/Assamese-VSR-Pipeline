"""
Upload already-processed results from temp directory to Google Drive
without reprocessing any videos.
"""
import os
import sys
from pathlib import Path
from drive_batch_pipeline import DriveManager

def upload_saved():
    temp_dir = Path(os.environ["TEMP"]) / "vsr_drive_gb60_jvo" / "vsr_results"
    if not temp_dir.exists():
        print(f"[ERROR] Temp folder not found: {temp_dir}")
        return

    print(f"[INFO] Found saved results in: {temp_dir}")
    dm = DriveManager("credentials.json")

    dest_path = "vsr_results"
    print(f"[INFO] Resolving destination path in your personal Google Drive: {dest_path}...")
    dest_id = dm.resolve_path(dest_path, create=True)
    print(f"[OK] Destination folder ID in your Drive: {dest_id}")

    for sub in ["landmarks", "graphs", "mouth_crops", "metadata", "previews"]:
        sub_dir = temp_dir / sub
        if sub_dir.exists() and any(sub_dir.iterdir()):
            print(f"\n[UPLOAD] Uploading {sub}/ to Google Drive...")
            sub_id = dm.get_or_create_folder(sub, dest_id)
            dm.upload_directory(str(sub_dir), sub_id)

    print("\n" + "="*50)
    print("[SUCCESS] All 50 processed videos uploaded to Google Drive!")
    print("="*50)

if __name__ == "__main__":
    upload_saved()

