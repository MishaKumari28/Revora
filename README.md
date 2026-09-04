# AI-Powered Revenue Recovery & Customer Retention System

Predict which subscription customers are likely to leave, estimate **revenue at risk**, score how recoverable that revenue looks, and recommend a retention action — then explore it in a Streamlit dashboard.

> **Interview-ready note:** Churn is an ML prediction. Revenue-at-risk and recoverable revenue are **estimates**. Recovery probability is a **business heuristic** because this dataset has **no historical recovery labels**.

---

## Problem statement

Subscription businesses lose revenue when customers churn. A model that only says “will churn / will not churn” is not enough for operations. This project answers:

1. Who is likely to churn?
2. How much near-term revenue is exposed?
3. Which accounts are worth a recovery attempt?
4. What action should the team take (billing, support, engagement, offer)?

## Motivation

For a final-year CSE / placement portfolio, the project demonstrates a full pipeline: data inspection, EDA, leakage-aware preprocessing, model comparison (not accuracy-only), a business layer, explainability, and a usable dashboard.

---

## Features

- Automatic dataset profiling (shape, types, missing values, duplicates, outliers)
- EDA with Plotly (and Matplotlib/Seaborn in the notebook)
- Scikit-learn `Pipeline` + `ColumnTransformer` (no preprocessing leakage)
- Logistic Regression, Decision Tree, Random Forest, XGBoost comparison
- Model selection by **Recall + ROC-AUC** (missing a churner costs revenue)
- Expected revenue at risk with a documented 6-month horizon
- Rule-based recovery recommendations
- Optional LLM recovery messages via `.env` (never hard-coded keys)
- Streamlit dashboard: overview, risk table, analytics, customer detail, simulator, explainability
- Customer-level contributing-factor text (not causal claims)

---

## Architecture

```
CSV → clean/profile → EDA → feature engineering → train/test split
    → sklearn Pipeline (impute / scale or not / one-hot / classifier)
    → churn probability
    → EAR = monthly_fee × 6 × P(churn)
    → recovery score (heuristic) + recommended action
    → Streamlit dashboard / optional LLM message
```

---

## Technology stack

| Layer | Tools |
|---|---|
| Data | Pandas, NumPy |
| EDA | Matplotlib, Seaborn, Plotly |
| ML | Scikit-learn, XGBoost, Joblib |
| Explainability | Tree feature importance, optional SHAP |
| App | Streamlit |
| LLM (optional) | OpenAI API via `.env` |

---

## Dataset description

Source file (copied into `data/`):

`data/customer_subscription_churn_usage_patterns.csv`

Inspected (do not assume other Kaggle schemas):

| Item | Value |
|---|---|
| Rows | 2,800 |
| Columns | 10 |
| Missing values | 0 |
| Duplicate rows | 0 |
| IQR outliers | 0 on numeric behaviour fields |
| Target | `churn` (`Yes` / `No`) |
| Observed churn rate | ~57.3% (`Yes` = 1,605) |
| Plans | Basic ₹199, Standard ₹399, Premium ₹699 |

**Source columns:** `user_id`, `signup_date`, `plan_type`, `monthly_fee`, `avg_weekly_usage_hours`, `support_tickets`, `payment_failures`, `tenure_months`, `last_login_days_ago`, `churn`

Full descriptions: [`data/DATA_DICTIONARY.md`](data/DATA_DICTIONARY.md)

Stronger raw associations with churn in this file: **payment failures**, **days since last login**, **support tickets**. Weekly usage is **negatively** associated with churn.

---

## ML methodology

- Target: `churn == "Yes"` → 1
- Features: the six numeric behaviour/price fields, `plan_type`, plus same-row ratios (`tickets_per_tenure_month`, `failures_per_tenure_month`, `usage_to_fee_ratio`)
- `user_id` is excluded
- `signup_date` is parsed for EDA; tenure already represents longevity in the model
- Stratified 80/20 train/test split, `random_state=42`
- Logistic Regression uses scaled numerics; tree models do not (unnecessary)
- Class weighting / `scale_pos_weight` for imbalance handling
- **Selection score** = `0.45 * recall + 0.40 * ROC-AUC + 0.15 * F1`

### Model Comparison Benchmark Results

Evaluated on a held-out test split (20% stratified, `random_state=42`) with feature engineering and leak-free `ColumnTransformer` preprocessing:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Selection Score |
|---|---|---|---|---|---|---|
| 🏆 **XGBoost** (Production Winner) | **66.79%** | **70.27%** | **72.90%** | **71.56%** | **72.47%** | **0.7253** |
| Random Forest Classifier | 65.36% | 70.96% | 66.98% | 68.91% | 72.38% | 0.6943 |
| Logistic Regression (Balanced) | 65.36% | 72.28% | 64.17% | 67.99% | 70.30% | 0.6720 |
| Decision Tree Classifier | 62.32% | 73.50% | 53.58% | 61.98% | 71.43% | 0.6198 |

**Optimization Rationale:**
In subscription revenue recovery, a **False Negative** (failing to identify an account about to churn) causes unrecoverable lifetime revenue loss. Therefore, model selection prioritizes **Recall and ROC-AUC**:
$$\text{Selection Score} = 0.45 \times \text{Recall} + 0.40 \times \text{ROC-AUC} + 0.15 \times \text{F1}$$

---

## Revenue-Risk & Recovery Methodology

### 1. Expected Revenue at Risk (EAR)
**Documented Assumption:** The CSV does not contain contract expiration dates. Rather than fabricating dates, we establish a transparent, documented 6-month prospective forecast horizon:

$$\text{Expected Revenue at Risk} = \text{Monthly Fee} \times 6 \times P(\text{Churn})$$

### 2. Risk Classification Bands
Customers are stratified by churn probability into operational tiers:

| Risk Tier | Churn Probability Range | Strategic Urgency |
|---|---|---|
| **Low Risk** | $< 35\%$ | Healthy / standard lifecycle nurture |
| **Medium Risk** | $35\% - 60\%$ | Early friction / engagement warning |
| **High Risk** | $60\% - 80\%$ | Active churn hazard / targeted outreach |
| **Critical Risk** | $\ge 80\%$ | Imminent churn / executive intervention |

### 3. Recoverability Probability Score
Because the dataset does not contain historical recovery experiment labels, recoverability is modeled as an **explainable behavioral heuristic**:
- Starts at **72% base potential**
- Deductions for payment failures (-10% each), high support tickets (-4% each), and platform dormancy (>14 days: -12%, >30 days: -20%)
- Bonuses for high weekly engagement (+1% per hr) and subscriber tenure (>12 mos: +8%)
- Bounded strictly between **5% and 95%**

$$\text{Estimated Recoverable Revenue} = \text{Expected Revenue at Risk} \times \text{Recovery Probability}$$


---

## Screenshots

Add dashboard captures here for GitHub / the report:

1. Overview KPIs — `assets/overview.png`
2. Customer risk table — `assets/risk_table.png`
3. Analytics charts — `assets/analytics.png`
4. Customer detail + message — `assets/customer_detail.png`
5. Simulator — `assets/simulator.png`

---

## Installation (Windows)

From the project root (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Optional LLM:

```powershell
copy .env.example .env
# then edit .env and set OPENAI_API_KEY
```

---

## How to run

Train models and score every customer:

```powershell
python -m src.train_model
```

Launch the dashboard:

```powershell
streamlit run app.py
```

Open the notebook:

```powershell
jupyter notebook notebooks/EDA_and_Modeling.ipynb
```

---

## Folder structure

```
AI-Revenue-Recovery/
├── data/
│   ├── customer_subscription_churn_usage_patterns.csv
│   └── DATA_DICTIONARY.md
├── models/                  # created by training
│   ├── churn_model.pkl
│   ├── preprocessing.pkl
│   ├── model_comparison.csv
│   └── feature_importance.csv
├── notebooks/
│   └── EDA_and_Modeling.ipynb
├── src/
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── train_model.py
│   ├── revenue_risk.py
│   ├── recovery_engine.py
│   ├── explainability.py
│   ├── llm_messages.py
│   └── scoring.py
├── assets/                  # screenshot placeholders
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

### What each module does

| Module | Role |
|---|---|
| `data_preprocessing.py` | Load, profile, clean, same-row features, sklearn preprocessor |
| `eda.py` | KPI helpers and Plotly figures |
| `train_model.py` | Train four classifiers, pick a winner, save artifacts, score the CSV |
| `revenue_risk.py` | EAR formula and risk bands |
| `recovery_engine.py` | Recovery score + action rules |
| `explainability.py` | Feature importance + customer narratives |
| `llm_messages.py` | Template or OpenAI message |
| `scoring.py` | Load artifacts for the app |
| `app.py` | Streamlit UI |

---

## Limitations

- No recovery / campaign-response labels, so recovery probability is **not** a trained historical rate.
- Remaining subscription months are unknown; EAR uses a fixed horizon.
- Churn labels in the file are historical; production use would need a time-aware split (train on past, predict future).
- LLM messages are optional and must not include extra PII.
- Feature importance and SHAP are associations, not causes.

## Disclaimer

Revenue at risk, recoverable revenue, and recovery probability are **predictions or estimates** for analysis and demonstration. They are not audited financial figures and should not be used as the sole basis for customer treatment without a human review process.

---

## Future improvements

- Time-based validation and production monitoring (data drift)
- Uplift modeling if campaign experiments become available
- Cost-sensitive thresholds (false negative vs discount cost)
- SHAP waterfall plots on the customer page
- Auth, audit logs, and role-based views for a real ops team
- A/B test of recommended actions

---

## Placement Interview & Final-Year Viva Preparation Guide

Here are standard questions interviewers and examiners ask regarding this system, along with model answers:

### Q1: Why did you prioritize Recall over Accuracy in model selection?
> **Answer:** In subscription businesses, prediction errors have highly asymmetric business costs. A **False Positive** (predicting churn for a customer who would have stayed) only costs a light nurture email or retention discount offer. However, a **False Negative** (failing to identify an account about to churn) results in permanent, unrecoverable revenue loss (e.g., ₹699/month × 6 months = ₹4,194 per customer). By prioritizing **Recall and ROC-AUC**, our selection function `0.45 * Recall + 0.40 * ROC-AUC + 0.15 * F1` ensures maximum revenue capture.

### Q2: How did you prevent Data Leakage across preprocessing and feature engineering?
> **Answer:** We prevented leakage at two critical junctures:
> 1. **Feature Engineering Isolation:** All engineered features (`tickets_per_tenure_month`, `usage_to_fee_ratio`, etc.) are purely row-level behavioral metrics. No cross-row aggregate statistics (like dataset mean or standard deviation) were calculated before splitting.
> 2. **Pipeline Preprocessing:** We employed Scikit-learn's `ColumnTransformer` inside an isolated pipeline. Scaling parameters (`mean_`, `var_`) and imputation medians were fitted **exclusively on the training fold** and merely transformed on the test fold.

### Q3: Why is "Recovery Probability" scored via heuristics rather than supervised learning?
> **Answer:** The source dataset does not contain historical retention campaign intervention outcomes (e.g., whether an offered discount succeeded in retaining a customer). Claiming to train a supervised model without target ground-truth labels would be scientifically fraudulent. Instead, we explicitly framed recoverability as an explainable, behavior-driven business scoring model based on domain logic (penalizing payment failures and inactivity while rewarding tenure and usage).

### Q4: How do you address the distinction between Model Correlation and Causality?
> **Answer:** Machine learning models identify statistical associations, not definitive causal links. In our explainability module and dashboard, we deliberately avoid causal assertions (e.g., we do *not* say "High support tickets caused churn"). Instead, we state: *"Contributing risk factors: 3 support tickets, 18 days inactivity."* This ensures ethical AI compliance and prevents misinformed business interventions.

---

## License / Academic Use

Built as a high-impact B.Tech CSE Final-Year Major Project, portfolio demonstration, and placement interview piece.

