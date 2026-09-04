"""Revenue-at-Risk calculation, dynamic subscription horizons, and customer Lifetime Value (LTV) engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.config import FORECAST_HORIZON_MONTHS, RISK_BANDS


def compute_dynamic_horizon(tenure_months_series: pd.Series, default_horizon: int = FORECAST_HORIZON_MONTHS) -> pd.Series:
    """Estimates realistic forward subscription contract horizon based on customer tenure maturity.
    
    Enterprise SaaS Cohort Dynamics:
    - Early lifecycle (tenure <= 4 months): 4-month conservative horizon (high volatility)
    - Mid lifecycle (5 to 18 months): 6-month standard forecast horizon
    - Established lifecycle (tenure > 18 months): 8-month stable renewal horizon
    """
    conditions = [
        tenure_months_series <= 4,
        (tenure_months_series > 4) & (tenure_months_series <= 18),
        tenure_months_series > 18,
    ]
    choices = [4, 6, 8]
    return pd.Series(np.select(conditions, choices, default=default_horizon), index=tenure_months_series.index)


def classify_risk_vectorized(churn_probs: np.ndarray) -> np.ndarray:
    """Vectorized classification of churn probabilities into operational risk tiers (0 ms overhead)."""
    conditions = [
        churn_probs < 0.35,
        (churn_probs >= 0.35) & (churn_probs < 0.60),
        (churn_probs >= 0.60) & (churn_probs < 0.80),
        churn_probs >= 0.80,
    ]
    choices = ["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]
    return np.select(conditions, choices, default="Medium Risk")


def classify_risk_category(prob: float) -> str:
    """Scalar risk classifier for backward compatibility."""
    if prob < 0.35:
        return "Low Risk"
    elif prob < 0.60:
        return "Medium Risk"
    elif prob < 0.80:
        return "High Risk"
    else:
        return "Critical Risk"


def calculate_revenue_at_risk(
    df: pd.DataFrame,
    churn_probs: np.ndarray | pd.Series,
    forecast_horizon_months: int = FORECAST_HORIZON_MONTHS,
    use_dynamic_horizon: bool = True,
) -> pd.DataFrame:
    """Calculates financial exposure and expected remaining LTV using fast vectorized math.
    
    Formulas:
    - Expected Revenue at Risk = Monthly Fee × Remaining Horizon × P(Churn)
    - Expected Remaining LTV = Monthly Fee × Remaining Horizon × (1 - P(Churn))
    """
    df_calc = df.copy()
    probs = np.asarray(churn_probs, dtype=float)
    df_calc["churn_probability"] = np.round(probs, 4)

    if use_dynamic_horizon and "tenure_months" in df_calc.columns:
        df_calc["estimated_remaining_months"] = compute_dynamic_horizon(
            df_calc["tenure_months"], default_horizon=forecast_horizon_months
        )
    else:
        df_calc["estimated_remaining_months"] = int(forecast_horizon_months)

    # Vectorized revenue calculations
    fee = df_calc["monthly_fee"].to_numpy(dtype=float)
    horizon = df_calc["estimated_remaining_months"].to_numpy(dtype=float)
    p_churn = df_calc["churn_probability"].to_numpy(dtype=float)

    revenue_risk = np.round(fee * horizon * p_churn, 2)
    remaining_ltv = np.round(fee * horizon * (1.0 - p_churn), 2)

    df_calc["revenue_at_risk"] = revenue_risk
    df_calc["expected_revenue_at_risk"] = revenue_risk
    df_calc["expected_remaining_ltv"] = remaining_ltv

    # Vectorized risk categorization
    risk_labels = classify_risk_vectorized(p_churn)
    df_calc["risk_level"] = risk_labels
    df_calc["risk_category"] = risk_labels

    return df_calc