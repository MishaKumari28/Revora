"""Project-wide configuration, paths, random seed, and documented business assumptions."""

from pathlib import Path

# Reproducibility
RANDOM_SEED = 42
TEST_SIZE = 0.20

# Directory paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "customer_subscription_churn_usage_patterns.csv"
PROCESSED_DIR = DATA_DIR / "processed"
SCORED_DATA_PATH = PROCESSED_DIR / "scored_customers.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Model artifact paths
CHURN_MODEL_PATH = MODELS_DIR / "churn_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessing.pkl"
METRICS_PATH = MODELS_DIR / "model_comparison.csv"
FEATURE_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.csv"

# Dataset column identifiers
ID_COLUMN = "user_id"
DATE_COLUMN = "signup_date"
TARGET_COLUMN = "churn"
POSITIVE_LABEL = "Yes"

CATEGORICAL_FEATURES = ["plan_type"]
NUMERIC_FEATURES = [
    "monthly_fee",
    "avg_weekly_usage_hours",
    "support_tickets",
    "payment_failures",
    "tenure_months",
    "last_login_days_ago",
]
DERIVED_NUMERIC_FEATURES = [
    "tickets_per_tenure_month",
    "failures_per_tenure_month",
    "usage_to_fee_ratio",
    "inactivity_ratio",
]

# Forecast horizon for Expected Revenue at Risk:
# Note: Actual remaining contract duration is not provided in the dataset.
# We establish a documented business forecast horizon assumption of 6 months.
FORECAST_HORIZON_MONTHS = 6

# Granular risk categories
RISK_BANDS = {
    "Low Risk": (0.0, 0.35),
    "Medium Risk": (0.35, 0.60),
    "High Risk": (0.60, 0.80),
    "Critical Risk": (0.80, 1.00),
}

# Recovery score clipping bounds (business heuristic)
RECOVERY_SCORE_BOUNDS = (0.05, 0.95)

