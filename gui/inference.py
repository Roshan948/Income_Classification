"""
Standalone inference for the income classifier.

Loads the single fitted `Pipeline` produced at the end of
`income_classification.ipynb` (preprocessing + classifier bundled together)
and exposes a `predict()` function that takes a plain dict of raw field
values and returns a label plus a probability.

Unlike a setup with separate encoder/scaler objects, there's no manual
column-by-column re-encoding here — the pipeline was fit on a dataframe with
these exact column names, so predicting on a new record is just building a
one-row dataframe with the same columns and calling `.predict_proba`.
"""

from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

PIPELINE = joblib.load(ARTIFACT_DIR / "pipeline.joblib")
METADATA = joblib.load(ARTIFACT_DIR / "metadata.joblib")

EDUCATION_MAP = METADATA["education_map"]
FEATURE_COLS = METADATA["feature_cols"]


def _build_row(raw: dict) -> pd.DataFrame:
    """Turn a dict of raw field values into the one-row dataframe the
    pipeline expects. Raises KeyError with a clear message if a field is
    missing rather than silently defaulting."""

    missing = [c for c in ("age", "capital_gain", "capital_loss", "hours_per_week",
                            "workclass", "education", "marital_status", "occupation",
                            "relationship", "race", "gender", "native_country")
               if c not in raw]
    if missing:
        raise KeyError(f"Missing input field(s): {missing}")

    education_band = EDUCATION_MAP.get(raw["education"], raw["education"])

    row = {
        "workclass": raw["workclass"],
        "marital-status": raw["marital_status"],
        "occupation": raw["occupation"],
        "relationship": raw["relationship"],
        "race": raw["race"],
        "gender": raw["gender"],
        "native-country": raw["native_country"],
        "education_band": education_band,
        "age": raw["age"],
        "capital-gain": raw["capital_gain"],
        "capital-loss": raw["capital_loss"],
        "hours-per-week": raw["hours_per_week"],
    }

    return pd.DataFrame([row], columns=FEATURE_COLS)


def predict(raw: dict) -> tuple[int, float]:
    """
    raw: dict with numeric keys age, capital_gain, capital_loss,
    hours_per_week, and categorical keys workclass, education,
    marital_status, occupation, relationship, race, gender, native_country
    (raw string values matching the categories seen at training time —
    unseen categories are handled gracefully thanks to
    handle_unknown="ignore" on the one-hot step).

    Returns (predicted_label, probability_of_over_50k).
    """

    row_df = _build_row(raw)
    prediction = int(PIPELINE.predict(row_df)[0])
    probability = float(PIPELINE.predict_proba(row_df)[0][1])
    return prediction, probability


if __name__ == "__main__":
    sample = {
        "age": 41,
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 45,
        "workclass": "Private",
        "education": "Bachelors",
        "marital_status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "gender": "Female",
        "native_country": "United-States",
    }

    label, prob = predict(sample)
    print("Prediction:", ">50K" if label == 1 else "<=50K", f"({prob:.1%} confidence)")
