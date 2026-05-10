import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from layer2.pipeline import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CHARTA Layer 2 — NLP Extraction")
    parser.add_argument("--input",   default="data/processed")
    parser.add_argument("--output",  default="data/extracted")
    parser.add_argument("--mode",    default="run",
                        choices=["run", "finetune", "eval"])
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()

    if args.mode == "run":
        run_pipeline(args.input, args.output)
    elif args.mode == "finetune":
        from layer2.trainer import finetune_relation_model  # created in Step 34
        finetune_relation_model(dataset_name=args.dataset or "bigbio/bc5cdr")
    elif args.mode == "eval":
        from layer2.evaluator import evaluate_ner
        evaluate_ner(args.dataset or "ncbi/ncbi_disease")
