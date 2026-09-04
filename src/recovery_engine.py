"""Recovery probability scoring, behavior-driven retention recommendation engine, and modular AI outreach."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.config import RECOVERY_SCORE_BOUNDS

load_dotenv()


def calculate_recovery_probability(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates heuristic likelihood score that an at-risk customer can be retained.
    
    IMPORTANT BUSINESS NOTICE:
    This score is a documented heuristic business assumption based on behavioral signals.
    Because the Kaggle subscription dataset contains no historical outreach/recovery intervention labels,
    we explicitly distinguish this business heuristic from the ML churn prediction.
    
    Behavioral Factors:
    - Base potential: 0.72 (72%)
    - Payment failures: penalty of -0.10 per failure (billing friction)
    - Support tickets: penalty of -0.04 per ticket (frustration/unresolved issues)
    - Inactivity: penalty of -0.12 if inactive > 14 days, -0.20 if inactive > 30 days
    - Usage engagement: bonus of +0.01 per weekly usage hour (habitual usage)
    - Tenure loyalty: bonus of +0.08 if tenure > 12 months (established relationship)
    - High churn probability dampener: up to -0.15 for extreme churn probabilities
    """
    df_res = df.copy()

    # Base recovery potential
    recovery_score = pd.Series(0.72, index=df_res.index)

    # Negative friction penalties
    recovery_score -= (df_res["payment_failures"] * 0.10)
    recovery_score -= (df_res["support_tickets"] * 0.04)
    recovery_score -= np.where(df_res["last_login_days_ago"] > 30, 0.20,
                              np.where(df_res["last_login_days_ago"] > 14, 0.12, 0.0))

    if "churn_probability" in df_res.columns:
        recovery_score -= np.where(df_res["churn_probability"] > 0.75, 0.12, 0.0)

    # Positive engagement & relationship bonuses
    recovery_score += (df_res["avg_weekly_usage_hours"] * 0.01)
    recovery_score += np.where(df_res["tenure_months"] > 12, 0.08, 0.0)

    # Clip to realistic business bounds (e.g. 5% to 95%)
    min_bound, max_bound = RECOVERY_SCORE_BOUNDS
    df_res["recovery_probability"] = np.clip(recovery_score, min_bound, max_bound).round(2)

    # Estimated potential recovered revenue
    rev_risk_col = "revenue_at_risk" if "revenue_at_risk" in df_res.columns else "expected_revenue_at_risk"
    if rev_risk_col in df_res.columns:
        df_res["estimated_recoverable_revenue"] = (
            df_res[rev_risk_col] * df_res["recovery_probability"]
        ).round(2)

    return df_res


def assign_recovery_actions_vectorized(df: pd.DataFrame) -> pd.Series:
    """Vectorized assignment of behavior-driven retention strategies (executes in <1ms)."""
    risk = df.get("risk_level", df.get("risk_category", pd.Series("", index=df.index)))
    payment_fails = df.get("payment_failures", pd.Series(0, index=df.index)).fillna(0).astype(int)
    monthly_fee = df.get("monthly_fee", pd.Series(0.0, index=df.index)).fillna(0).astype(float)
    usage = df.get("avg_weekly_usage_hours", pd.Series(0.0, index=df.index)).fillna(0).astype(float)
    tickets = df.get("support_tickets", pd.Series(0, index=df.index)).fillna(0).astype(int)
    inactivity = df.get("last_login_days_ago", pd.Series(0, index=df.index)).fillna(0).astype(int)

    conditions = [
        risk == "Low Risk",
        payment_fails >= 2,
        (payment_fails == 1) & (monthly_fee >= 399),
        payment_fails == 1,
        tickets >= 3,
        (usage < 6.0) | (inactivity > 14),
        monthly_fee >= 699,
    ]

    choices = [
        "No action",
        "Payment retry & Concierge billing call",
        "Payment reminder & Priority billing assistance",
        "Payment reminder",
        "Customer support follow-up",
        "Engagement campaign",
        "Discount/offer recommendation",
    ]

    actions = np.select(conditions, choices, default="Personalized retention message")
    return pd.Series(actions, index=df.index)


def assign_recovery_action(row: pd.Series | Dict[str, Any]) -> str:
    """Scalar backward-compatible action mapper."""
    risk = str(row.get("risk_level", row.get("risk_category", "")))
    if risk == "Low Risk":
        return "No action"

    payment_fails = int(row.get("payment_failures", 0) or 0)
    monthly_fee = float(row.get("monthly_fee", 0) or 0)
    usage = float(row.get("avg_weekly_usage_hours", 0) or 0)
    tickets = int(row.get("support_tickets", 0) or 0)
    inactivity = int(row.get("last_login_days_ago", 0) or 0)

    if payment_fails >= 2:
        return "Payment retry & Concierge billing call"
    elif payment_fails == 1:
        if monthly_fee >= 399:
            return "Payment reminder & Priority billing assistance"
        return "Payment reminder"

    if tickets >= 3:
        return "Customer support follow-up"

    if usage < 6.0 or inactivity > 14:
        return "Engagement campaign"

    if monthly_fee >= 699:
        return "Discount/offer recommendation"

    return "Personalized retention message"


def compute_intervention_economics(df: pd.DataFrame) -> pd.DataFrame:
    """Incorporates authentic SaaS unit economics: intervention cost, net recoverable value, and ROI multiples."""
    out = df.copy()

    if "recommended_action" not in out.columns:
        out["recommended_action"] = assign_recovery_actions_vectorized(out)

    action_series = out["recommended_action"]
    fee = out.get("monthly_fee", pd.Series(0.0, index=out.index)).astype(float)
    horizon = out.get("estimated_remaining_months", pd.Series(6.0, index=out.index)).astype(float)

    # Realistic enterprise intervention costs
    # - Discount offer cost: 20% price concession over the remaining horizon
    discount_cost = (0.20 * fee * horizon).round(2)

    cost_conditions = [
        action_series == "No action",
        action_series == "Payment reminder",
        action_series == "Payment reminder & Priority billing assistance",
        action_series == "Payment retry & Concierge billing call",
        action_series == "Customer support follow-up",
        action_series == "Engagement campaign",
        action_series == "Discount/offer recommendation",
    ]

    cost_choices = [
        0.0,            # Zero cost
        15.0,           # Automated SMS/Email retry gateway fee
        35.0,           # Billing ops ticket routing fee
        65.0,           # Senior billing concierge specialist triage
        120.0,          # Technical solutions engineering call
        10.0,           # In-app product guide and automated email
        discount_cost,  # 20% commercial concession
    ]

    cost = np.select(cost_conditions, cost_choices, default=25.0)
    out["intervention_cost"] = np.round(cost, 2)

    # Net financial yield after intervention costs
    rec_rev = out.get("estimated_recoverable_revenue", pd.Series(0.0, index=out.index)).astype(float)
    net_val = (rec_rev - out["intervention_cost"]).round(2)
    out["net_recoverable_value"] = net_val

    # ROI multiple: Gross Recovered / Intervention Cost
    safe_cost = np.maximum(out["intervention_cost"].to_numpy(dtype=float), 1.0)
    out["roi_multiple"] = np.round(rec_rev.to_numpy(dtype=float) / safe_cost, 1)

    # Flag economically viable targets
    out["is_profitable_intervention"] = (net_val > 0) & (action_series != "No action")

    return out



def generate_fallback_template_message(row: pd.Series | Dict[str, Any]) -> str:
    """Creates a high-quality, professional recovery message without requiring external API calls."""
    plan = str(row.get("plan_type", "Subscription"))
    action = str(row.get("recommended_action", "Personalized retention message"))
    tenure = int(row.get("tenure_months", 0) or 0)
    usage = float(row.get("avg_weekly_usage_hours", 0) or 0)
    fails = int(row.get("payment_failures", 0) or 0)
    tickets = int(row.get("support_tickets", 0) or 0)

    if "Payment" in action:
        body = (
            f"We noticed a recent processing issue regarding your {plan} plan subscription. "
            f"To make sure your platform access continues smoothly without interruption, "
            f"we have placed a temporary 5-day grace period on your account. "
            f"Please take a moment to update your billing details or let our concierge billing team assist you."
        )
        cta = "Update Billing Information ->"
    elif "Customer support" in action:
        body = (
            f"Thank you for being a valued {plan} subscriber for {tenure} months. "
            f"We noticed you recently logged {tickets} support ticket(s). Our senior technical success lead "
            f"would love to connect with you directly for a 10-minute check-in to ensure all issues are resolved to your satisfaction."
        )
        cta = "Schedule 10-Min Support Call ->"
    elif "Engagement" in action:
        body = (
            f"We hope you are enjoying your {plan} plan. "
            f"We noticed your activity has been quieter recently ({usage:.1f} hrs/week). "
            f"Our team put together a personalized 3-minute feature walkthrough covering new workflow automations "
            f"designed to save you 4+ hours every week."
        )
        cta = "Explore Feature Walkthrough ->"
    elif "Discount" in action:
        body = (
            f"As one of our premier {plan} members with {tenure} months on the platform, your partnership is vital to us. "
            f"To thank you for your ongoing loyalty, we are pleased to extend an exclusive 20% annual loyalty renewal discount "
            f"applied automatically to your next quarterly billing cycle."
        )
        cta = "Claim 20% Loyalty Credit ->"
    else:
        body = (
            f"Thank you for being a dedicated {plan} subscriber for the past {tenure} months. "
            f"We are continuously releasing enhancements based on customer feedback and would love to ensure "
            f"you have everything you need to achieve your goals."
        )
        cta = "View Product Updates ->"


    message = (
        f"Subject: Important update regarding your {plan} subscription\n\n"
        f"Hi there,\n\n"
        f"{body}\n\n"
        f"Action: {cta}\n\n"
        f"Warm regards,\n"
        f"The Customer Success & Retention Team"
    )
    return message


def generate_llm_recovery_message(
    customer_row: pd.Series | Dict[str, Any],
    api_key: Optional[str] = None,
    custom_instructions: str = "",
) -> str:
    """Generates personalized recovery outreach using an LLM (OpenAI) with graceful local fallback.
    
    Privacy Guarantee:
    - Never transmits sensitive PII (passwords, emails, physical addresses, or financial credentials).
    - Only summarizes non-identifying behavioral attributes (plan, usage, tickets, recommended strategy).
    """
    effective_api_key = api_key if api_key else os.getenv("OPENAI_API_KEY", "").strip()

    if not effective_api_key:
        return generate_fallback_template_message(customer_row)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=effective_api_key)

        prompt = f"""
You are a senior customer retention specialist for a subscription SaaS company.
Draft a personalized, empathetic, concise email (under 110 words) for a customer with this behavioral profile:

- Plan Tier: {customer_row.get('plan_type', 'Standard')}
- Customer Tenure: {customer_row.get('tenure_months', 0)} months
- Weekly Platform Usage: {customer_row.get('avg_weekly_usage_hours', 0)} hours
- Recent Payment Failures: {customer_row.get('payment_failures', 0)}
- Support Ticket Count: {customer_row.get('support_tickets', 0)}
- Days Since Last Login: {customer_row.get('last_login_days_ago', 0)}
- Primary Retention Strategy: {customer_row.get('recommended_action', 'Personalized retention')}
{f'- Specific Instructions: {custom_instructions}' if custom_instructions else ''}

Rules:
1. Do NOT mention internal churn probabilities, risk scores, or numeric algorithms.
2. Tone must be helpful, respectful, and focused on value delivery.
3. Include a clear, low-friction call-to-action button or link.
"""

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You craft concise, high-converting customer retention messages."},
                {"role": "user", "content": prompt.strip()},
            ],
            max_tokens=220,
            temperature=0.6,
        )
        content = response.choices[0].message.content
        return content.strip() if content else generate_fallback_template_message(customer_row)

    except Exception as e:
        fallback = generate_fallback_template_message(customer_row)
        return f"[Notice: Live LLM call encountered: {str(e)}. Using system template]\n\n{fallback}"