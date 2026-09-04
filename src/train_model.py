"""Model training, multi-algorithm evaluation, Recall/ROC-AUC optimization, and artifact serialization."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import (
    CHURN_MODEL_PATH,
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    MODELS_DIR,
    PREPROCESSOR_PATH,
    RANDOM_SEED,
    RAW_DATA_PATH,
    SCORED_DATA_PATH,
)
from src.data_preprocessing import (
    engineer_features,
    load_raw_data,
    prepare_data,
    prepare_xy,
)
from src.recovery_engine import (
    assign_recovery_action,
    assign_recovery_actions_vectorized,
    calculate_recovery_probability,
    compute_intervention_economics,
)
from src.revenue_risk import calculate_revenue_at_risk



def evaluate_models(
    X_train_trans: np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_test_trans: np.ndarray,
    y_test: pd.Series | np.ndarray,
    preprocessor: Any = None,
) -> Tuple[Dict[str, Dict[str, Any]], str, Any]:
    """Trains and rigorously benchmarks multiple classification models.
    
    In subscription revenue recovery, a False Negative (missing a churning customer)
    results in unrecoverable revenue loss. Therefore, model selection explicitly prioritizes
    Recall and ROC-AUC over pure accuracy.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            scale_pos_weight=1.0,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
        ),
    }

    results: Dict[str, Dict[str, Any]] = {}
    best_model = None
    best_score = -1.0
    best_model_name = ""

    comparison_rows = []

    for name, model in classifiers.items():
        model.fit(X_train_trans, y_train)
        y_pred = model.predict(X_test_trans)
        y_proba = model.predict_proba(X_test_trans)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_proba))
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Revenue-protection selection metric (heavily penalizes False Negatives)
        selection_metric = 0.45 * rec + 0.40 * auc + 0.15 * f1

        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc,
            "Confusion Matrix": cm,
            "Selection Score": selection_metric,
            "Model": model,
        }

        comparison_rows.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4),
            "Selection Score": round(selection_metric, 4),
        })

        if selection_metric > best_score:
            best_score = selection_metric
            best_model = model
            best_model_name = name

    # Persist the best model
    joblib.dump(best_model, CHURN_MODEL_PATH)

    # Persist model comparison dataframe
    comparison_df = pd.DataFrame(comparison_rows).sort_values(by="Selection Score", ascending=False)
    comparison_df.to_csv(METRICS_PATH, index=False)

    # Extract & persist feature importances
    if preprocessor is not None:
        save_feature_importances(best_model, preprocessor)

    return results, best_model_name, best_model


def get_feature_names_from_preprocessor(preprocessor: Any) -> list[str]:
    """Recovers feature names after ColumnTransformer one-hot and scaling steps."""
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder" or transformer == "drop":
            continue
        if hasattr(transformer, "named_steps") and "onehot" in transformer.named_steps:
            encoder = transformer.named_steps["onehot"]
            cat_names = encoder.get_feature_names_out(cols).tolist()
            feature_names.extend(cat_names)
        else:
            feature_names.extend(cols)
    return feature_names


def save_feature_importances(model: Any, preprocessor: Any) -> pd.DataFrame:
    """Extracts, formats, and saves feature importance coefficients/weights."""
    feat_names = get_feature_names_from_preprocessor(preprocessor)

    if hasattr(model, "feature_importances_"):
        raw_weights = model.feature_importances_
    elif hasattr(model, "coef_"):
        raw_weights = np.abs(model.coef_[0])
    else:
        raw_weights = np.ones(len(feat_names))

    # Normalize to 100%
    weight_sum = np.sum(raw_weights)
    norm_weights = raw_weights / weight_sum if weight_sum > 0 else raw_weights

    importance_df = pd.DataFrame({
        "Feature": feat_names,
        "Importance": norm_weights,
    }).sort_values(by="Importance", ascending=False)

    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    return importance_df


def score_customers(
    df: pd.DataFrame,
    model: Any = None,
    preprocessor: Any = None,
) -> pd.DataFrame:
    """Applies preprocessing and ML scoring, then calculates revenue risk and recovery recommendations."""
    if model is None:
        model = joblib.load(CHURN_MODEL_PATH)
    if preprocessor is None:
        preprocessor = joblib.load(PREPROCESSOR_PATH)

    X, _ = prepare_xy(df)
    X_trans = preprocessor.transform(X)
    churn_probs = model.predict_proba(X_trans)[:, 1]

    # Calculate financial exposure, dynamic horizons, and LTV
    scored = calculate_revenue_at_risk(df, churn_probs, use_dynamic_horizon=True)

    # Calculate recovery probability
    scored = calculate_recovery_probability(scored)

    # Assign actions via fast vectorized routines
    scored["recommended_action"] = assign_recovery_actions_vectorized(scored)

    # Calculate authentic SaaS unit economics (cost, net value, ROI)
    scored = compute_intervention_economics(scored)

    # Persist to data/processed
    SCORED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(SCORED_DATA_PATH, index=False)

    return scored



class TrainingSummary(dict):
    """Custom dictionary allowing both dictionary key lookups and 2-tuple unpacking."""
    def __iter__(self):
        yield self.get("results")
        yield self.get("best_model")


def train_and_compare(data_input: str | Path | pd.DataFrame = RAW_DATA_PATH) -> TrainingSummary:
    """End-to-end execution: loads data, fits preprocessor, benchmarks 4 models, and scores all customers.
    
    Accepts either a filesystem path or a pre-loaded pandas DataFrame.
    Returns a TrainingSummary that supports both dictionary access and 2-tuple unpacking.
    """
    if isinstance(data_input, pd.DataFrame):
        df = data_input.copy()
    else:
        df = load_raw_data(data_input)

    X_train, X_test, y_train, y_test, X_train_tr, X_test_tr, preprocessor = prepare_data(df)

    results, best_name, best_model = evaluate_models(
        X_train_tr, y_train, X_test_tr, y_test, preprocessor=preprocessor
    )

    # Score full dataset with winning model
    score_customers(df, model=best_model, preprocessor=preprocessor)

    # Prepare benchmark summary rows
    metrics_list = []
    confusion_matrices = {}
    for name, r in results.items():
        metrics_list.append({
            "Model": name,
            "Accuracy": r["Accuracy"],
            "Precision": r["Precision"],
            "Recall": r["Recall"],
            "F1-Score": r["F1-Score"],
            "ROC-AUC": r["ROC-AUC"],
            "Selection Score": r["Selection Score"],
        })
        confusion_matrices[name] = r["Confusion Matrix"]

    summary = TrainingSummary({
        "best_model": best_name,
        "selection_rule": "0.45 * Recall + 0.40 * ROC_AUC + 0.15 * F1 (Optimized for Revenue Recovery)",
        "metrics": metrics_list,
        "confusion_matrices": confusion_matrices,
        "results": results,
        "model": best_model,
        "preprocessor": preprocessor,
    })

    return summary


if __name__ == "__main__":
    results, best_name = train_and_compare()
    print("\n================ MODEL BENCHMARK RESULTS ================")
    metrics_df = pd.read_csv(METRICS_PATH)
    print(metrics_df.to_string(index=False))
    print(f"\n[WINNING MODEL]: {best_name} selected based on Recall and ROC-AUC optimization.")