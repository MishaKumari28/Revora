"""Data loading, automated dataset profiling, feature engineering, and leak-free preprocessing pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    DATE_COLUMN,
    DERIVED_NUMERIC_FEATURES,
    ID_COLUMN,
    NUMERIC_FEATURES,
    POSITIVE_LABEL,
    PREPROCESSOR_PATH,
    RANDOM_SEED,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
)

# Comprehensive Data Dictionary
DATA_DICTIONARY: Dict[str, str] = {
    "user_id": "Unique integer identifier for each subscription customer.",
    "signup_date": "Original registration date (YYYY-MM-DD); captures cohort timing.",
    "plan_type": "Subscription tier: 'Basic' (₹199), 'Standard' (₹399), or 'Premium' (₹699).",
    "monthly_fee": "Recurring monthly price billed in INR (₹). Aligns with plan tier.",
    "avg_weekly_usage_hours": "Average hours the customer spends actively utilizing the platform per week.",
    "support_tickets": "Total count of customer service / technical support tickets lodged.",
    "payment_failures": "Count of failed billing or payment transactions recorded.",
    "tenure_months": "Active duration of the customer subscription in months.",
    "last_login_days_ago": "Recency indicator: count of days since customer last authenticated.",
    "churn": "Ground truth target variable: 'Yes' (customer churned), 'No' (customer retained).",
    "tickets_per_tenure_month": "Engineered feature: Support ticket rate per active tenure month.",
    "failures_per_tenure_month": "Engineered feature: Payment failure rate per active tenure month.",
    "usage_to_fee_ratio": "Engineered feature: Engagement value delivered per unit cost (hours/₹).",
    "inactivity_ratio": "Engineered feature: Days since last login normalized by total subscription lifespan.",
}

# Aliases for compatibility
NUMERICAL_FEATURES = NUMERIC_FEATURES


@dataclass
class DatasetProfile:
    """Statistical and structural metadata summary of the dataset."""
    n_rows: int
    n_columns: int
    dtypes: Dict[str, str]
    missing_values: Dict[str, int]
    duplicate_rows: int
    nunique: Dict[str, int]
    numeric_columns: List[str]
    categorical_columns: List[str]
    date_columns: List[str]
    target_column: str
    outlier_counts_iqr: Dict[str, int]
    summary_stats: Dict[str, Any] = field(default_factory=dict)


def load_raw_data(filepath: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Loads CSV dataset with fallback checks."""
    path = Path(filepath)
    if not path.exists():
        fallback = Path("customer_subscription_churn_usage_patterns.csv")
        if fallback.exists():
            path = fallback
        else:
            raise FileNotFoundError(f"Dataset not found at {filepath} or local directory.")
    return pd.read_csv(path)


# Backward-compatible alias
load_data = load_raw_data


def profile_dataset(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> DatasetProfile:
    """Automatically profiles dataset structure, completeness, and distributions."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Identify potential date columns
    date_cols: List[str] = []
    for col in categorical_cols:
        if "date" in col.lower() or "time" in col.lower():
            date_cols.append(col)

    # Detect outliers using 1.5 * IQR standard rule
    outlier_counts: Dict[str, int] = {}
    for col in numeric_cols:
        if col == ID_COLUMN:
            continue
        series = df[col].dropna()
        q25, q75 = np.percentile(series, [25, 75])
        iqr = q75 - q25
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr
        outliers = int(((series < lower) | (series > upper)).sum())
        outlier_counts[col] = outliers

    return DatasetProfile(
        n_rows=int(df.shape[0]),
        n_columns=int(df.shape[1]),
        dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
        missing_values=df.isnull().sum().to_dict(),
        duplicate_rows=int(df.duplicated().sum()),
        nunique=df.nunique().to_dict(),
        numeric_columns=[c for c in numeric_cols if c != ID_COLUMN],
        categorical_columns=[c for c in categorical_cols if c not in date_cols and c != target_col],
        date_columns=date_cols,
        target_column=target_col if target_col in df.columns else "",
        outlier_counts_iqr=outlier_counts,
        summary_stats=df.describe().to_dict(),
    )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes explainable behavioral domain ratios without cross-row leakage."""
    out = df.copy()

    # Denominator smoothing with np.maximum to prevent division by zero
    tenure_safe = np.maximum(out["tenure_months"].astype(float), 1.0)
    fee_safe = np.maximum(out["monthly_fee"].astype(float), 1.0)

    # Ticket frequency over subscription lifespan
    out["tickets_per_tenure_month"] = (out["support_tickets"] / tenure_safe).round(4)

    # Failure density over subscription lifespan
    out["failures_per_tenure_month"] = (out["payment_failures"] / tenure_safe).round(4)

    # Economic utility ratio: usage hours received per Rupee spent
    out["usage_to_fee_ratio"] = (out["avg_weekly_usage_hours"] / fee_safe).round(4)

    # Inactivity ratio relative to total customer tenure days
    out["inactivity_ratio"] = (out["last_login_days_ago"] / (tenure_safe * 30.0)).round(4)

    return out


def encode_target(series: pd.Series) -> pd.Series:
    """Encodes churn column: 'Yes' -> 1, 'No' -> 0."""
    if series.dtype == object or str(series.dtype) == "category":
        return (series.astype(str).str.strip().str.capitalize() == POSITIVE_LABEL).astype(int)
    return series.astype(int)


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Creates a robust Scikit-Learn ColumnTransformer pipeline.
    
    Scales numerical variables and one-hot encodes categorical variables.
    Fitted exclusively on training data to strictly prevent data leakage.
    """
    all_numeric_features = NUMERIC_FEATURES + DERIVED_NUMERIC_FEATURES

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, all_numeric_features),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Applies feature engineering and separates features X from target y."""
    engineered = engineer_features(df)
    feature_cols = NUMERIC_FEATURES + DERIVED_NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X = engineered[feature_cols].copy()
    y = encode_target(df[TARGET_COLUMN]) if TARGET_COLUMN in df.columns else pd.Series()
    return X, y


def prepare_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray, np.ndarray, ColumnTransformer]:
    """Performs stratified train/test split and fits preprocessing on train only."""
    X, y = prepare_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessing_pipeline()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Persist the fitted preprocessor
    PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    return X_train, X_test, y_train, y_test, X_train_trans, X_test_trans, preprocessor