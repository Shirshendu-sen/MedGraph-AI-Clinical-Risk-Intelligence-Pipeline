"""
run_pipeline.py — CHARTA full end-to-end pipeline
Processes a folder of plain text clinical notes through all 5 layers.
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from layer1.pipeline import run_pipeline as l1
from layer2.pipeline import run_pipeline as l2
from layer3.pipeline import run_pipeline as l3
from layer4.pipeline import run_pipeline as l4
from layer5.pipeline import run_pipeline as l5

def run_full_pipeline(input_folder: str) -> None:
    start = time.time()
    print(f"\n{'='*60}")
    print("CHARTA Full Pipeline Starting")
    print(f"Input: {input_folder}")
    print(f"{'='*60}\n")

    print("[Layer 1] Clinical Text Preprocessing...")
    l1(input_folder, "data/processed")

    print("[Layer 2] Clinical NLP Extraction...")
    l2("data/processed", "data/extracted")

    print("[Layer 3] Temporal Patient Graph...")
    l3("data/extracted", "data/graphs")

    print("[Layer 4] Readmission Risk Prediction...")
    l4("data/graphs", "data/predictions")

    print("[Layer 5] Explainable Clinical Report...")
    l5("data/predictions", "data/graphs", "data/explanations")

    elapsed = round(time.time() - start, 1)
    print(f"\n{'='*60}")
    print(f"Pipeline complete in {elapsed}s")
    print(f"Reports in: data/explanations/")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CHARTA End-to-End Pipeline")
    parser.add_argument("--input", default="data/raw/txt", help="Folder of plain text clinical notes")
    args = parser.parse_args()
    run_full_pipeline(args.input)