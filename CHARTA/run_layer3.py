import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from layer3.pipeline import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CHARTA Layer 3 — Temporal Graph")
    parser.add_argument("--input",  default="data/extracted")
    parser.add_argument("--output", default="data/graphs")
    args = parser.parse_args()
    run_pipeline(args.input, args.output)
