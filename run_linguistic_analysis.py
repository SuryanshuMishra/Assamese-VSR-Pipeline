"""
CLI Runner for Automated Linguistic C-V and Boundary Analysis
============================================================
Supports single-sentence processing and full-corpus batch processing.

Usage:
  # Single folder:
  python run_linguistic_analysis.py --input s320_1351_output --lab path/to/transcript.lab

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
    parser = argparse.ArgumentParser(description="Automated Linguistic C-V & Boundary Analysis for Assamese LMR")
    parser.add_argument("--input", type=str, help="Path to a single processed sentence output folder")
    parser.add_argument("--lab", type=str, default=None, help="Path to ground truth .lab transcript file (optional)")
    parser.add_argument("--batch", type=str, help="Directory containing multiple processed sentence folders")
    parser.add_argument("--output-summary", type=str, default="corpus_linguistic_summary.csv", help="Master summary CSV for batch mode")
    args = parser.parse_args()

    segmenter = LinguisticSegmenter()

    if args.input:
        in_path = Path(args.input)
        if not in_path.exists():
            print(f"Error: Path {in_path} does not exist.")
            sys.exit(1)

        print(f"[*] Processing sentence folder: {in_path}")
        summary = segmenter.process_sentence(in_path, lab_file=args.lab)
        print("\n=== LINGUISTIC SUMMARY ===")
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
            # Check direct subdirectories
            subdirs = [p for p in batch_dir.iterdir() if p.is_dir() and (p / "csvs" / "shape_features.csv").exists()]

        print(f"[*] Found {len(subdirs)} processed sentence folders in {batch_dir}")
        all_summaries = []

        for folder in subdirs:
            print(f"  -> Processing: {folder.name}")
            # Check if there is an adjacent or matching .lab file
            matching_lab = list(folder.glob("*.lab"))
            lab_path = matching_lab[0] if matching_lab else None

            try:
                summary = segmenter.process_sentence(folder, lab_file=lab_path)
                all_summaries.append(summary)
            except Exception as e:
                print(f"     [!] Failed on {folder.name}: {e}")

        if all_summaries:
            summary_df = pd.DataFrame(all_summaries)
            out_file = Path(args.output_summary)
            summary_df.to_csv(out_file, index=False)
            print(f"\n[OK] Successfully processed {len(all_summaries)} sentences.")
            print(f"[OK] Master corpus summary saved to: {out_file}")
            print(summary_df.to_string())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

