"""CHARTA Layer 5 — Explainable Clinical Report Generation entry point.

Usage:
    python run_layer5.py --predictions data/predictions --graphs data/graphs --output data/explanations
"""

import argparse
import sys
from pathlib import Path

# Ensure CHARTA/src is on the Python path for Colab compatibility
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CHARTA Layer 5 — Explainable Clinical Report"
    )
    parser.add_argument(
        "--predictions",
        default="data/predictions",
        help="Folder containing *_predictions.json from Layer 4",
    )
    parser.add_argument(
        "--graphs",
        default="data/graphs",
        help="Folder containing *_graph.pt and *_graph_meta.json from Layer 3",
    )
    parser.add_argument(
        "--output",
        default="data/explanations",
        help="Output folder for *_report.json files",
    )
    args = parser.parse_args()

    from layer5.pipeline import run_pipeline

    run_pipeline(args.predictions, args.graphs, args.output)