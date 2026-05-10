"""Prepare MTSamples dataset: read CSV, write individual .txt files."""

import argparse
import pathlib
import re

import pandas as pd


def read_mtsamples_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    df = df[df["transcription"].notna() & (df["transcription"].str.strip() != "")]
    return df.reset_index(drop=True)


def make_slug(sample_name: str) -> str:
    # No slugify library needed — use built-in str methods only
    return re.sub(r"[^\w]", "_", sample_name.lower())[:60]


def write_transcription_files(df: pd.DataFrame, output_dir: str) -> None:
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    for idx, row in df.iterrows():
        slug = make_slug(str(row["sample_name"]))
        fname = f"mtsamples_{idx:04d}_{slug}.txt"
        pathlib.Path(output_dir, fname).write_text(
            row["transcription"], encoding="utf-8", errors="replace"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MTSamples .txt files")
    parser.add_argument("--input", default="data/mtsamples/mtsamples.csv")
    parser.add_argument("--output", default="data/raw/txt/mtsamples/")
    args = parser.parse_args()

    df = read_mtsamples_csv(args.input)
    print(f"Loaded {len(df)} transcriptions from {args.input}")
    write_transcription_files(df, args.output)
    print(f"Wrote .txt files to {args.output}")
