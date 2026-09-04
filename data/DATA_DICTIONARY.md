# Data dictionary — source CSV plus calculated fields

## Source columns (from `customer_subscription_churn_usage_patterns.csv`)

| Column | Type in file | Role | Description |
|---|---|---|---|
| user_id | integer | ID | Unique customer identifier. Not used as a model feature. |
| signup_date | date string | EDA | Subscription start date (`YYYY-MM-DD`). Tenure already captures longevity for modeling. |
| plan_type | category | Feature | `Basic`, `Standard`, or `Premium`. |
| monthly_fee | integer | Feature / revenue | Monthly price in INR. In this file: 199 / 399 / 699 aligned with plan. |
| avg_weekly_usage_hours | float | Feature | Average weekly product usage hours. |
| support_tickets | integer | Feature | Count of support tickets. |
| payment_failures | integer | Feature | Count of failed payments. |
| tenure_months | integer | Feature | Months subscribed so far. |
| last_login_days_ago | integer | Feature | Days since last login. Higher means less recent activity. |
| churn | category | Target | `Yes` = churned, `No` = retained. |

## Derived / estimated fields (not in the original CSV)

| Column | Kind | Description |
|---|---|---|
| tickets_per_tenure_month | derived, same-row | `support_tickets / tenure_months` |
| failures_per_tenure_month | derived, same-row | `payment_failures / tenure_months` |
| usage_to_fee_ratio | derived, same-row | `avg_weekly_usage_hours / monthly_fee` |
| churn_probability | ML prediction | Probability of churn from the saved sklearn pipeline |
| expected_revenue_at_risk | estimate | `monthly_fee × 6 × churn_probability` |
| risk_category | calculated | Low / Medium / High / Critical from probability bands |
| recovery_probability | business score | Heuristic recoverability; **no historical recovery labels exist** |
| recommended_action | rule-based | Suggested ops / retention action |
| estimated_recoverable_revenue | estimate | `expected_revenue_at_risk × recovery_probability` |
