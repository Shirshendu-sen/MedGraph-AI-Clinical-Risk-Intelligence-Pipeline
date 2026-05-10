from datasets import load_dataset
from pathlib import Path

# Load OpenI dataset from Hugging Face cache
ds = load_dataset("ykumards/open-i")

# Output directory
out_dir = Path("data/raw/txt/openI")
out_dir.mkdir(parents=True, exist_ok=True)

count = 0

for idx, sample in enumerate(ds["train"]):

    findings = sample.get("findings", "")
    impression = sample.get("impression", "")
    indication = sample.get("indication", "")
    problems = sample.get("Problems", "")
    mesh = sample.get("MeSH", "")

    report = f"""
UID: {sample.get('uid')}

INDICATION:
{indication}

FINDINGS:
{findings}

IMPRESSION:
{impression}

PROBLEMS:
{problems}

MESH:
{mesh}
"""

    with open(out_dir / f"openi_{idx:04d}.txt", "w", encoding="utf-8") as f:
        f.write(report.strip())

    count += 1

print(f"Exported {count} reports to {out_dir}")
