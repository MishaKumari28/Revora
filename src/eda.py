"""Exploratory analysis helpers used by the notebook and Streamlit analytics."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import TARGET_COLUMN
from src.data_preprocessing import encode_target


PLOTLY_TEMPLATE = "plotly_white"
COLOR_CHURN = {"Yes": "#d62728", "No": "#2ca02c"}


def with_churn_flag(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["churn_flag"] = encode_target(out[TARGET_COLUMN])
    return out


def kpi_summary(df: pd.DataFrame) -> dict[str, float]:
    work = with_churn_flag(df)
    return {
        "n_customers": int(len(work)),
        "churn_rate": float(work["churn_flag"].mean()),
        "total_monthly_revenue": float(work["monthly_fee"].sum()),
        "churned_monthly_revenue": float(work.loc[work["churn_flag"] == 1, "monthly_fee"].sum()),
    }


def churn_by_plan(df: pd.DataFrame) -> pd.DataFrame:
    work = with_churn_flag(df)
    table = (
        work.groupby("plan_type", as_index=False)
        .agg(
            customers=("user_id", "count"),
            churn_rate=("churn_flag", "mean"),
            monthly_revenue=("monthly_fee", "sum"),
        )
        .sort_values("plan_type")
    )
    return table


def fig_churn_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df[TARGET_COLUMN].value_counts().reset_index()
    counts.columns = ["churn", "count"]
    fig = px.pie(
        counts,
        names="churn",
        values="count",
        color="churn",
        color_discrete_map=COLOR_CHURN,
        title="Churn distribution",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(textinfo="label+percent+value")
    return fig


def fig_monthly_revenue(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="monthly_fee",
        color=TARGET_COLUMN,
        barmode="group",
        color_discrete_map=COLOR_CHURN,
        title="Monthly fee distribution by churn",
        template=PLOTLY_TEMPLATE,
        nbins=10,
    )
    fig.update_layout(xaxis_title="Monthly fee (INR)", yaxis_title="Customers")
    return fig


def fig_plan_churn(df: pd.DataFrame) -> go.Figure:
    table = churn_by_plan(df)
    fig = px.bar(
        table,
        x="plan_type",
        y="churn_rate",
        text=table["churn_rate"].map(lambda x: f"{x:.1%}"),
        title="Plan-wise churn rate",
        template=PLOTLY_TEMPLATE,
        color="plan_type",
    )
    fig.update_layout(yaxis_tickformat=".0%", showlegend=False)
    fig.update_traces(textposition="outside")
    return fig


def fig_payment_failures_vs_churn(df: pd.DataFrame) -> go.Figure:
    work = with_churn_flag(df)
    rates = work.groupby("payment_failures", as_index=False)["churn_flag"].mean()
    fig = px.bar(
        rates,
        x="payment_failures",
        y="churn_flag",
        title="Payment failures vs churn rate",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(yaxis_title="Churn rate", yaxis_tickformat=".0%")
    return fig


def fig_support_vs_churn(df: pd.DataFrame) -> go.Figure:
    work = with_churn_flag(df)
    rates = work.groupby("support_tickets", as_index=False)["churn_flag"].mean()
    fig = px.bar(
        rates,
        x="support_tickets",
        y="churn_flag",
        title="Support tickets vs churn rate",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(yaxis_title="Churn rate", yaxis_tickformat=".0%")
    return fig


def fig_usage_vs_churn(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df,
        x=TARGET_COLUMN,
        y="avg_weekly_usage_hours",
        color=TARGET_COLUMN,
        color_discrete_map=COLOR_CHURN,
        title="Weekly usage hours vs churn",
        template=PLOTLY_TEMPLATE,
    )
    return fig


def fig_tenure_vs_churn(df: pd.DataFrame) -> go.Figure:
    fig = px.violin(
        df,
        x=TARGET_COLUMN,
        y="tenure_months",
        color=TARGET_COLUMN,
        color_discrete_map=COLOR_CHURN,
        box=True,
        title="Tenure vs churn",
        template=PLOTLY_TEMPLATE,
    )
    return fig


def fig_last_login_vs_churn(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df,
        x=TARGET_COLUMN,
        y="last_login_days_ago",
        color=TARGET_COLUMN,
        color_discrete_map=COLOR_CHURN,
        title="Days since last login vs churn",
        template=PLOTLY_TEMPLATE,
    )
    return fig


def fig_correlation(df: pd.DataFrame) -> go.Figure:
    work = with_churn_flag(df)
    cols = [
        "monthly_fee",
        "avg_weekly_usage_hours",
        "support_tickets",
        "payment_failures",
        "tenure_months",
        "last_login_days_ago",
        "churn_flag",
    ]
    corr = work[cols].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation between numerical variables",
        template=PLOTLY_TEMPLATE,
    )
    return fig


def fig_revenue_by_plan(df: pd.DataFrame) -> go.Figure:
    table = (
        df.groupby("plan_type", as_index=False)["monthly_fee"]
        .sum()
        .rename(columns={"monthly_fee": "monthly_revenue"})
    )
    fig = px.pie(
        table,
        names="plan_type",
        values="monthly_revenue",
        title="Monthly revenue contribution by plan",
        template=PLOTLY_TEMPLATE,
    )
    return fig


def fig_high_risk_groups(df: pd.DataFrame) -> go.Figure:
    work = with_churn_flag(df)
    work["usage_band"] = pd.cut(
        work["avg_weekly_usage_hours"],
        bins=[0, 6, 13, 20, 40],
        labels=["Very low", "Low", "Medium", "High"],
        include_lowest=True,
    )
    grouped = (
        work.groupby(["plan_type", "usage_band"], observed=False)["churn_flag"]
        .mean()
        .reset_index()
    )
    fig = px.bar(
        grouped,
        x="usage_band",
        y="churn_flag",
        color="plan_type",
        barmode="group",
        title="Churn rate by plan and usage band (high-risk groups)",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Churn rate")
    return fig


def save_static_eda_plots(df: pd.DataFrame, output_dir: str = "reports/figures") -> list[str]:
    """Generates and saves publication-quality static Matplotlib/Seaborn figures for reports and presentations."""
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    sns.set_theme(style="whitegrid", font_scale=1.1)

    work = with_churn_flag(df)

    # 1. Churn Distribution
    plt.figure(figsize=(6, 5))
    churn_counts = df[TARGET_COLUMN].value_counts()
    plt.pie(churn_counts, labels=churn_counts.index, autopct="%1.1f%%", colors=["#e74c3c", "#2ecc71"], startangle=90)
    plt.title("Overall Customer Churn Distribution", fontsize=14, fontweight="bold")
    path1 = os.path.join(output_dir, "eda_churn_distribution.png")
    plt.tight_layout()
    plt.savefig(path1, dpi=300)
    plt.close()
    saved_paths.append(path1)

    # 2. Correlation Matrix Heatmap
    plt.figure(figsize=(9, 7))
    cols = [
        "monthly_fee", "avg_weekly_usage_hours", "support_tickets",
        "payment_failures", "tenure_months", "last_login_days_ago", "churn_flag"
    ]
    corr = work[cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Correlation Matrix of Behavioral Features", fontsize=14, fontweight="bold")
    path2 = os.path.join(output_dir, "eda_correlation_matrix.png")
    plt.tight_layout()
    plt.savefig(path2, dpi=300)
    plt.close()
    saved_paths.append(path2)

    # 3. Payment Failures & Support Tickets vs Churn
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=work, x="payment_failures", y="churn_flag", ax=axes[0], color="#e74c3c", errorbar=None)
    axes[0].set_title("Payment Failures vs Churn Rate", fontweight="bold")
    axes[0].set_ylabel("Observed Churn Rate")
    axes[0].set_ylim(0, 1)

    sns.barplot(data=work, x="support_tickets", y="churn_flag", ax=axes[1], color="#3498db", errorbar=None)
    axes[1].set_title("Support Tickets vs Churn Rate", fontweight="bold")
    axes[1].set_ylabel("Observed Churn Rate")
    axes[1].set_ylim(0, 1)

    path3 = os.path.join(output_dir, "eda_failures_and_tickets_vs_churn.png")
    plt.tight_layout()
    plt.savefig(path3, dpi=300)
    plt.close()
    saved_paths.append(path3)

    return saved_paths


