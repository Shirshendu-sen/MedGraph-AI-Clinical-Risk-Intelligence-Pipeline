"""Assemble final JSON report from top features."""

import json


def format_risk_level(prob: float) -> str:
    """Classify risk probability into 'LOW' (< 0.5) or 'HIGH' (>= 0.5)."""
    return "LOW" if prob < 0.5 else "HIGH"


def build_plain_english_summary(risk_score: float, top_features: list[dict]) -> str:
    """Template-based plain-English summary - no LLM required.

    Example output:
      "High readmission risk (84%) due to hypertension, diabetes, and multiple medications."
    """
    level = format_risk_level(risk_score)
    factor_names = ", ".join(f["entity_name"] for f in top_features)
    return (
        f"{'High' if level=='HIGH' else 'Low'} readmission risk "
        f"({risk_score * 100:.0f}%) due to {factor_names}."
    )


def build_report(patient_id: str, prediction: dict, top_features: list[dict]) -> dict:
    """Assemble the final explainable clinical report as a JSON-serializable dict."""
    return {
        "metadata": {
            "patient_id": patient_id,
            "layer": "layer5_explainable_clinical_report",
        },
        "risk_summary": {
            "readmission_risk": prediction["readmission_risk"],
            "risk_level": prediction["risk_level"],
        },
        "explanation": {
            "plain_english": build_plain_english_summary(
                prediction["readmission_risk"], top_features
            ),
            "top_factors": top_features,  # uses concept_id, NOT cui
        },
        "disclaimer": "Research prototype - not a substitute for clinical judgment.",
    }


def save_report(report: dict, output_path: str) -> None:
    """Write the report dict to a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)