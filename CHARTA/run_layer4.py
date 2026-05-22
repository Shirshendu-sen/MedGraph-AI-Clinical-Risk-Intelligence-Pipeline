"""CHARTA Layer 4 — Readmission Risk Prediction entry point.

Usage:
    python run_layer4.py --mode train  [--epochs 20] [--batch 16]
    python run_layer4.py --mode eval   [--checkpoint models/best_readmission_model.pt]
    python run_layer4.py --mode run    [--input data/mtsamples_graphs --predictions data/predictions]

The --input flag accepts multiple folders for single-patient inference:
    python run_layer4.py --mode run --input data/mtsamples_graphs data/openI_graphs
"""

import argparse
import sys
from pathlib import Path

# Ensure CHARTA/src is on the Python path for Colab compatibility
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CHARTA Layer 4 — Readmission Risk Prediction"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "run"],
        default="run",
        help="Mode: train (GNN training), eval (test metrics), run (single-patient inference)",
    )
    parser.add_argument(
        "--graphs",
        nargs="+",
        default=["data/mtsamples_graphs", "data/openI_graphs"],
        help="Graph folders for training/eval (default: data/mtsamples_graphs data/openI_graphs)",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=["data/mtsamples_graphs", "data/openI_graphs"],
        help="Graph folders for inference (default: data/mtsamples_graphs data/openI_graphs)",
    )
    parser.add_argument(
        "--predictions",
        default="data/predictions",
        help="Output folder for prediction JSONs (--mode run)",
    )
    parser.add_argument(
        "--checkpoint",
        default="models/best_readmission_model.pt",
        help="Path to model checkpoint (--mode eval/run)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override NUM_EPOCHS (--mode train)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Override BATCH_SIZE (--mode train/eval)",
    )
    parser.add_argument(
        "--labels",
        default="data/corpus_labels.csv",
        help="Path to corpus_labels.csv",
    )
    args = parser.parse_args()

    # ── Mode dispatch ────────────────────────────────────────────────
    if args.mode == "train":
        from layer4.trainer import train

        config_override = {
            "graphs_folder": args.graphs,
            "labels_csv": args.labels,
        }
        if args.epochs is not None:
            config_override["num_epochs"] = args.epochs
        if args.batch is not None:
            config_override["batch_size"] = args.batch

        train(config_override)

    elif args.mode == "eval":
        from layer4.trainer import evaluate_on_test

        evaluate_on_test(
            checkpoint_path=args.checkpoint,
            graphs_folder=args.graphs,
            labels_csv=args.labels,
            batch_size=args.batch or 16,
        )

    elif args.mode == "run":
        from layer4.pipeline import run_pipeline

        run_pipeline(
            input_folder=args.input,
            output_folder=args.predictions,
            checkpoint_path=args.checkpoint,
        )