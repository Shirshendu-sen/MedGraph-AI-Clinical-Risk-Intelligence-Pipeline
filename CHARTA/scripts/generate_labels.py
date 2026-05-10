import pandas as pd
import re
import argparse

HIGH_RISK_SPECIALTIES = {
    "Emergency Room Reports", "Cardiovascular / Pulmonary",
    "Nephrology", "Critical Care / Intensive Care"
}
READMIT_KEYWORDS   = ["readmit", "re-admit", "return to ed", "follow-up in 24 hours"]
DETERIORATE_WORDS  = ["acute","urgent","severe","critical","worsening",
                      "icu transfer","emergent","deteriorating"]
MEDICATION_WORDS   = ["drug interaction","polypharmacy","allergic to",
                      "adverse reaction","contraindicated"]

def derive_readmission(row) -> int:
    if row["medical_specialty"] in HIGH_RISK_SPECIALTIES: return 1
    t = str(row["transcription"]).lower()
    return int(any(k in t for k in READMIT_KEYWORDS))

def derive_deterioration(transcription: str) -> int:
    t = transcription.lower()
    return int(any(k in t for k in DETERIORATE_WORDS))

def derive_medication(transcription: str) -> int:
    t = transcription.lower()
    if any(k in t for k in MEDICATION_WORDS): return 1
    drug_tokens = re.findall(r"\b[A-Z][a-z]+(?:mab|nib|pril|statin|mycin|cillin)\b", transcription)
    return int(len(set(drug_tokens)) >= 5)

def run(mtsamples_csv, output_csv):
    df = pd.read_csv(mtsamples_csv, encoding="utf-8")
    df["medical_specialty"] = df["medical_specialty"].str.strip()
    df["patient_id"]    = [f"mtsamples_{i:04d}" for i in range(len(df))]
    df["readmission"]   = df.apply(derive_readmission, axis=1)
    df["deterioration"] = df["transcription"].fillna("").apply(derive_deterioration)
    df["medication"]    = df["transcription"].fillna("").apply(derive_medication)
    df[["patient_id","readmission","deterioration","medication"]].to_csv(output_csv, index=False)
    print(f"Labels saved: {output_csv}")
    print(df[["readmission","deterioration","medication"]].sum())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mtsamples", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.mtsamples, args.output)
