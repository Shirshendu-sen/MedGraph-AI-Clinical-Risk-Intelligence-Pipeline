import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from layer1.pipeline import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CHARTA Layer 1 — Document Ingestion")
    parser.add_argument("--input",  default="data/raw",       help="Input folder")
    parser.add_argument("--output", default="data/processed", help="Output folder")
    args = parser.parse_args()
    summary = run_pipeline(args.input, args.output)
    exit(0 if summary["failed"] == 0 else 1)
