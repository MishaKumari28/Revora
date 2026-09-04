"""Load persisted artifacts and score customers for the dashboard."""

from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

from src.config import CHURN_MODEL_PATH, PREPROCESSOR_PATH, SCORED_DATA_PATH
from src.data_preprocessing import load_raw_data
from src.train_model import score_customers, train_and_compare


def load_churn_model(path: Path = CHURN_MODEL_PATH):
    """Loads saved classification model."""
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact at {path}. Run: python -m src.train_model")
    return joblib.load(path)


def load_preprocessor(path: Path = PREPROCESSOR_PATH):
    """Loads saved ColumnTransformer preprocessor."""
    if not path.exists():
        raise FileNotFoundError(f"Missing preprocessor artifact at {path}. Run: python -m src.train_model")
    return joblib.load(path)


def load_or_build_scored_data(force_retrain: bool = False) -> pd.DataFrame:
    """Loads existing scored dataset or triggers full pipeline if artifacts are missing."""
    if (
        force_retrain
        or not SCORED_DATA_PATH.exists()
        or not CHURN_MODEL_PATH.exists()
        or not PREPROCESSOR_PATH.exists()
    ):
        train_and_compare()

    scored = pd.read_csv(SCORED_DATA_PATH)
    if "signup_date" in scored.columns:
        scored["signup_date"] = pd.to_datetime(scored["signup_date"], errors="coerce")
    return scored


def score_raw_if_needed() -> pd.DataFrame:
    """Convenience helper ensuring scored data is available."""
    if SCORED_DATA_PATH.exists() and CHURN_MODEL_PATH.exists() and PREPROCESSOR_PATH.exists():
        return load_or_build_scored_data()
    raw = load_raw_data()
    if CHURN_MODEL_PATH.exists() and PREPROCESSOR_PATH.exists():
        model = load_churn_model()
        prep = load_preprocessor()
        scored = score_customers(raw, model=model, preprocessor=prep)
        return scored
    return load_or_build_scored_data(force_retrain=True)

