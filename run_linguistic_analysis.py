"""
CLI Runner for Video-Only Linguistic C-V and Boundary Analysis
=============================================================
Purely video-based segmentation without any transcripts or audio files.

Usage:
  # Single processed folder:
  python run_linguistic_analysis.py --input test_s303/

  # Batch across all processed folders:
  python run_linguistic_analysis.py --batch results/ --output-summary corpus_linguistic_summary.csv
"""
import argparse
import sys
from pathlib import Path
import pandas as pd

from vlr.linguistics import LinguisticSegmenter

# Force UTF-8 stdout
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Video-Only Linguistic C-V & Boundary Analysis for Assamese LMR")
    parser.add_argument("--input", type=str, help="Path to a single processed video output folder")
    parser.add_argument("--batch", type=str, help="Directory containing multiple processed video folders")
    parser.add_argument("--output-summary", type=str, default="corpus_linguistic_summary.csv", help="Master summary CSV for batch mode")
    args = parser.parse_args()

    segmenter = LinguisticSegmenter()

    if args.input:
        in_path = Path(args.input)
        if not in_path.exists():
            print(f"Error: Path {in_path} does not exist.")
            sys.exit(1)

        print(f"[*] Processing video folder: {in_path}")
        summary = segmenter.process_sentence(in_path)
        print("\n=== VIDEO LINGUISTIC SUMMARY ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print(f"\n[OK] Results saved in {in_path / 'csvs'} and {in_path / 'graphs'}")

    elif args.batch:
        batch_dir = Path(args.batch)
        if not batch_dir.exists():
            print(f"Error: Batch directory {batch_dir} does not exist.")
            sys.exit(1)

        # Find all subdirectories containing shape_features.csv
        subdirs = [p.parent.parent for p in batch_dir.glob("**/shape_features.csv")]
        if not subdirs:
            subdirs = [p for p in batch_dir.iterdir() if p.is_dir() and (p / "csvs" / "shape_features.csv").exists()]

        print(f"[*] Found {len(subdirs)} processed video folders in {batch_dir}")
        all_summaries = []

        for folder in subdirs:
            print(f"  -> Processing: {folder.name}")
            try:
                summary = segmenter.process_sentence(folder)
                all_summaries.append(summary)
            except Exception as e:
                print(f"     [!] Failed on {folder.name}: {e}")

        if all_summaries:
            summary_df = pd.DataFrame(all_summaries)
            out_file = Path(args.output_summary)
            summary_df.to_csv(out_file, index=False)
            print(f"\n[OK] Successfully processed {len(all_summaries)} videos.")
            print(f"[OK] Master corpus summary saved to: {out_file}")
            print(summary_df.to_string())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
