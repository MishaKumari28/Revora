"""Optional LLM layer for personalized recovery messages.

Uses OPENAI_API_KEY from the environment / .env file. If the key is missing
or the API call fails, a template message is returned instead.
Never send unnecessary PII: we pass plan, risk, and behaviour aggregates only.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT


def load_dotenv_if_present() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
    except ImportError:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def template_recovery_message(row: pd.Series) -> str:
    plan = str(row.get("plan_type", "your"))
    action = str(row.get("recommended_action", "a retention follow-up"))
    tenure = int(row.get("tenure_months", 0) or 0)
    usage = float(row.get("avg_weekly_usage_hours", 0) or 0)
    fails = int(row.get("payment_failures", 0) or 0)
    risk = str(row.get("risk_category", "elevated"))

    if action in {"Payment reminder", "Payment retry"}:
        focus = (
            "We noticed a recent billing issue and want to help you restore access "
            "without losing your current plan benefits."
        )
    elif action == "Customer support follow-up":
        focus = (
            "We would like to resolve the issues behind your recent support tickets "
            "so the product works the way you expect."
        )
    elif action == "Engagement campaign":
        focus = (
            f"Your recent usage is around {usage:.1f} hours per week. "
            "We can share a short walkthrough of features that similar customers use most."
        )
    elif action == "Discount/offer recommendation":
        focus = (
            "Because you are on a higher-value plan, we can review a loyalty credit "
            "if it helps you stay with us this quarter."
        )
    elif action == "No action":
        focus = "Your account currently looks healthy. A light thank-you note is enough."
    else:
        focus = (
            "We would like to check in and make sure the subscription still matches "
            "how you use the product."
        )

    billing_note = (
        f" Our records show {fails} payment failure(s)." if fails else ""
    )
    return (
        f"Hi — thank you for {tenure} month(s) on the {plan} plan. "
        f"Your account is currently in the {risk} risk band.{billing_note} {focus} "
        f"Suggested next step for the team: {action}."
    )


def llm_recovery_message(row: pd.Series) -> str | None:
    load_dotenv_if_present()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    prompt = (
        "Write a short, professional customer-retention message (max 90 words). "
        "Do not include account numbers, emails, or any identifier. "
        f"Plan: {row.get('plan_type')}. "
        f"Risk band: {row.get('risk_category')}. "
        f"Tenure months: {row.get('tenure_months')}. "
        f"Weekly usage hours: {row.get('avg_weekly_usage_hours')}. "
        f"Payment failures: {row.get('payment_failures')}. "
        f"Support tickets: {row.get('support_tickets')}. "
        f"Days since last login: {row.get('last_login_days_ago')}. "
        f"Recommended internal action: {row.get('recommended_action')}. "
        "Tone: helpful, not pushy. End with one clear next step for the customer."
    )
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You write concise subscription retention emails."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=180,
        )
        text = response.choices[0].message.content
        return text.strip() if text else None
    except Exception:
        return None


def generate_recovery_message(row: pd.Series, use_llm: bool = True) -> tuple[str, str]:
    if use_llm:
        llm_text = llm_recovery_message(row)
        if llm_text:
            return llm_text, "llm"
    return template_recovery_message(row), "template"
