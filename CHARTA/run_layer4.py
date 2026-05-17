import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CHARTA Layer 4")
    parser.add_argument("--mode", choices=["build_index","train","eval","run"],
                        default="run")
    parser.add_argument("--graphs",  nargs="+", default=["data/mtsamples_graphs","data/openI_graphs"])
    parser.add_argument("--output",  default="data/corpus_index")
    parser.add_argument("--input",   default="data/graphs")
    parser.add_argument("--predictions", default="data/predictions")
    parser.add_argument("--graphs-folder", default="data/graphs")
    args = parser.parse_args()

    if args.mode == "build_index":
        from layer4.faiss_indexer import build_index_from_folders
        build_index_from_folders(args.graphs, args.output)
    elif args.mode == "train":
        from layer4.trainer import train
        train({"graphs_folder": args.graphs_folder})
    elif args.mode == "eval":
        from layer4.trainer import evaluate_on_test
        evaluate_on_test()
    elif args.mode == "run":
        from layer4.pipeline import run_pipeline
        run_pipeline(args.input, args.predictions)