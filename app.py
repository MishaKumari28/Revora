"""AI-Powered Revenue Recovery & Customer Retention System — Interactive Enterprise Production Dashboard."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.config import (
    CHURN_MODEL_PATH,
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    PREPROCESSOR_PATH,
    SCORED_DATA_PATH,
)
from src.data_preprocessing import (
    CATEGORICAL_FEATURES,
    DERIVED_NUMERIC_FEATURES,
    NUMERIC_FEATURES,
    load_raw_data,
)
from src.eda import (
    fig_churn_distribution,
    fig_correlation,
    fig_high_risk_groups,
    fig_last_login_vs_churn,
    fig_monthly_revenue,
    fig_payment_failures_vs_churn,
    fig_plan_churn,
    fig_revenue_by_plan,
    fig_support_vs_churn,
    fig_tenure_vs_churn,
    fig_usage_vs_churn,
)
from src.explainability import (
    explain_customer_risk,
    get_feature_importances,
    get_local_shap_explanation,
)
from src.recovery_engine import (
    assign_recovery_action,
    assign_recovery_actions_vectorized,
    calculate_recovery_probability,
    compute_intervention_economics,
    generate_llm_recovery_message,
)
from src.revenue_risk import calculate_revenue_at_risk
from src.scoring import load_or_build_scored_data

load_dotenv()

st.set_page_config(
    page_title="Enterprise AI Revenue Recovery System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for cards and badges
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .badge-profitable {
        background-color: #d4edda;
        color: #155724;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85em;
    }
    .badge-unprofitable {
        background-color: #f8d7da;
        color: #721c24;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_application_state():
    """Initializes and caches model, preprocessor, and scored customer dataset."""
    scored_df = load_or_build_scored_data()
    # Guarantee unit economics columns are present
    if "intervention_cost" not in scored_df.columns:
        scored_df = compute_intervention_economics(scored_df)
    model = joblib.load(CHURN_MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    raw_df = load_raw_data()
    return raw_df, scored_df, model, preprocessor


try:
    raw_df, scored_df, model, preprocessor = load_application_state()
except Exception as err:
    st.error(f"Failed to initialize application artifacts: {err}")
    st.info("Tip: Run `python -m src.train_model` in the terminal to build initial models.")
    st.stop()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=64)
st.sidebar.title("Revenue Recovery AI")
st.sidebar.caption("SaaS Retention & Unit-Economics Engine")

navigation_page = st.sidebar.radio(
    "Navigation Menu",
    [
        "Executive Overview",
        "Customer Risk Analysis",
        "Analytics & Behavioral EDA",
        "Customer Deep-Dive & AI Outreach",
        "Revenue Recovery Simulator",
        "Model Governance & Explainability",
    ],
)



# ==============================================================================
# 1. EXECUTIVE OVERVIEW
# ==============================================================================
if navigation_page == "Executive Overview":
    st.title("📊 Executive Revenue Protection Overview")
    st.markdown(
        "Executive-level overview balancing gross churn risk exposure with **retention intervention costs**, "
        "net recoverable margin, and portfolio ROI."
    )

    total_customers = len(scored_df)
    total_mrr = scored_df["monthly_fee"].sum()
    high_critical_df = scored_df[scored_df["risk_level"].isin(["High Risk", "Critical Risk"])]
    at_risk_count = len(high_critical_df)
    at_risk_pct = (at_risk_count / total_customers) * 100

    total_rev_at_risk = scored_df["revenue_at_risk"].sum()
    total_gross_recoverable = scored_df["estimated_recoverable_revenue"].sum()
    total_intervention_cost = scored_df[scored_df["risk_level"] != "Low Risk"]["intervention_cost"].sum()
    total_net_recoverable = scored_df[scored_df["risk_level"] != "Low Risk"]["net_recoverable_value"].sum()
    portfolio_roi = (total_gross_recoverable / max(total_intervention_cost, 1.0)) if total_intervention_cost > 0 else 0

    # Top KPI metric cards
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Active Customers", f"{total_customers:,}")
    k2.metric("Monthly MRR", f"₹{total_mrr:,.0f}")
    k3.metric("At-Risk Accounts", f"{at_risk_count:,}", f"{at_risk_pct:.1f}% of cohort", delta_color="inverse")
    k4.metric("Gross Revenue at Risk", f"₹{total_rev_at_risk:,.0f}", delta_color="inverse")
    k5.metric("Intervention Budget", f"₹{total_intervention_cost:,.0f}", "Required spend")
    k6.metric("Net Recoverable Margin", f"₹{total_net_recoverable:,.0f}", f"{portfolio_roi:.1f}x Net ROI")

    st.markdown("---")

    # High-level analytical charts
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Customer Distribution by Risk Tier")
        risk_summary = scored_df["risk_level"].value_counts().reset_index()
        risk_summary.columns = ["Risk Tier", "Customers"]
        fig_risk = px.pie(
            risk_summary,
            names="Risk Tier",
            values="Customers",
            color="Risk Tier",
            color_discrete_map={
                "Low Risk": "#2ecc71",
                "Medium Risk": "#f1c40f",
                "High Risk": "#e67e22",
                "Critical Risk": "#e74c3c",
            },
            hole=0.45,
        )
        fig_risk.update_traces(textinfo="label+percent+value")
        fig_risk.update_layout(margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_risk, use_container_width=True)

    with col_chart2:
        st.subheader("Net Recoverable Margin by Plan Tier (₹)")
        net_by_plan = scored_df.groupby("plan_type", as_index=False)["net_recoverable_value"].sum()
        fig_plan_rev = px.bar(
            net_by_plan,
            x="plan_type",
            y="net_recoverable_value",
            color="plan_type",
            color_discrete_sequence=["#3498db", "#9b59b6", "#2ecc71"],
            labels={"net_recoverable_value": "Net Recoverable Margin (₹)", "plan_type": "Subscription Plan"},
            text_auto=",.0f",
        )
        fig_plan_rev.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_plan_rev, use_container_width=True)

    st.markdown("---")
    st.subheader("⚡ Top 5 High-Yield Customer Accounts (Priority Focus Today)")
    top_accounts = scored_df.sort_values(by="net_recoverable_value", ascending=False).head(5)
    top_disp = top_accounts[[
        "user_id", "plan_type", "monthly_fee", "churn_probability",
        "revenue_at_risk", "intervention_cost", "net_recoverable_value", "roi_multiple", "recommended_action"
    ]].copy()
    top_disp["churn_probability"] = top_disp["churn_probability"].map("{:.1%}".format)
    top_disp["revenue_at_risk"] = top_disp["revenue_at_risk"].map("₹{:,.2f}".format)
    top_disp["intervention_cost"] = top_disp["intervention_cost"].map("₹{:,.2f}".format)
    top_disp["net_recoverable_value"] = top_disp["net_recoverable_value"].map("₹{:,.2f}".format)
    top_disp["roi_multiple"] = top_disp["roi_multiple"].map("{:.1f}x".format)
    st.dataframe(top_disp, use_container_width=True, hide_index=True)


# ==============================================================================
# 2. CUSTOMER RISK ANALYSIS (ACTION CENTER & BATCH DISPATCH)
# ==============================================================================
elif navigation_page == "Customer Risk Analysis":
    st.title("🎯 Customer Risk Analysis & Retention Action Center")
    st.markdown(
        "Cohort explorer with **unit-economics filtering** and one-click **Batch Campaign Dispatching**."
    )

    st.sidebar.subheader("Filter Customers")

    # Risk level filter
    all_risk_levels = ["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]
    existing_risk = [r for r in all_risk_levels if r in scored_df["risk_level"].unique()]
    selected_risk = st.sidebar.multiselect("Risk Tier", options=existing_risk, default=existing_risk)

    # Plan filter
    all_plans = scored_df["plan_type"].unique().tolist()
    selected_plans = st.sidebar.multiselect("Subscription Plan", options=all_plans, default=all_plans)

    # Churn probability slider
    prob_range = st.sidebar.slider("Churn Probability Range", 0.0, 1.0, (0.0, 1.0), step=0.05)

    # Payment failures filter
    max_fails = int(scored_df["payment_failures"].max())
    fail_range = st.sidebar.slider("Payment Failures Count", 0, max_fails, (0, max_fails))

    # Revenue at risk filter
    max_rev_risk = float(scored_df["revenue_at_risk"].max())
    min_rev_risk = st.sidebar.slider("Min. Revenue at Risk (₹)", 0.0, max_rev_risk, 0.0, step=100.0)

    # Realism filter: Only positive ROI
    only_profitable = st.sidebar.checkbox("Show Only Profitable Interventions (Net Value > 0)", value=False)

    # Apply filters
    filter_mask = (
        (scored_df["risk_level"].isin(selected_risk))
        & (scored_df["plan_type"].isin(selected_plans))
        & (scored_df["churn_probability"] >= prob_range[0])
        & (scored_df["churn_probability"] <= prob_range[1])
        & (scored_df["payment_failures"] >= fail_range[0])
        & (scored_df["payment_failures"] <= fail_range[1])
        & (scored_df["revenue_at_risk"] >= min_rev_risk)
    )

    if only_profitable and "is_profitable_intervention" in scored_df.columns:
        filter_mask = filter_mask & (scored_df["is_profitable_intervention"] == True)

    filtered_df = scored_df[filter_mask].copy()

    # Metrics on filtered population
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matching Customers", f"{len(filtered_df):,}")
    c2.metric("Gross Revenue at Risk", f"₹{filtered_df['revenue_at_risk'].sum():,.2f}")
    c3.metric("Required Spend", f"₹{filtered_df['intervention_cost'].sum():,.2f}")
    c4.metric("Net Recoverable Margin", f"₹{filtered_df['net_recoverable_value'].sum():,.2f}")

    display_cols = [
        "user_id",
        "plan_type",
        "monthly_fee",
        "tenure_months",
        "payment_failures",
        "churn_probability",
        "risk_level",
        "revenue_at_risk",
        "intervention_cost",
        "net_recoverable_value",
        "roi_multiple",
        "recommended_action",
    ]

    formatted_display = filtered_df[display_cols].sort_values(by="net_recoverable_value", ascending=False).copy()

    formatted_display["churn_probability"] = formatted_display["churn_probability"].map("{:.1%}".format)
    formatted_display["revenue_at_risk"] = formatted_display["revenue_at_risk"].map("₹{:,.2f}".format)
    formatted_display["intervention_cost"] = formatted_display["intervention_cost"].map("₹{:,.2f}".format)
    formatted_display["net_recoverable_value"] = formatted_display["net_recoverable_value"].map("₹{:,.2f}".format)
    formatted_display["monthly_fee"] = formatted_display["monthly_fee"].map("₹{:,.0f}".format)
    formatted_display["roi_multiple"] = formatted_display["roi_multiple"].map("{:.1f}x".format)

    st.dataframe(formatted_display, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🚀 Batch Retention Campaign Dispatcher")
    st.markdown(
        "Export operational webhook / CRM payloads (e.g. for HubSpot, Salesforce, or Stripe Billing retry workflows)."
    )

    b_col1, b_col2 = st.columns([1, 1])
    with b_col1:
        target_action = st.selectbox(
            "Select Target Intervention Strategy to Batch Dispatch:",
            options=[a for a in filtered_df["recommended_action"].unique() if a != "No action"],
        )
    with b_col2:
        dispatch_candidates = filtered_df[filtered_df["recommended_action"] == target_action]
        st.write(f"Target Queue: **{len(dispatch_candidates)}** customers ready for `{target_action}`.")

        # Prepare dispatch payload
        dispatch_export = dispatch_candidates[[
            "user_id", "plan_type", "monthly_fee", "tenure_months",
            "payment_failures", "churn_probability", "recommended_action",
            "intervention_cost", "net_recoverable_value"
        ]].copy()
        dispatch_export["dispatch_channel"] = np.where(
            dispatch_export["recommended_action"].str.contains("call", case=False), "Phone / VIP Concierge",
            np.where(dispatch_export["recommended_action"].str.contains("Payment", case=False), "Stripe Billing / SMS", "Customer Success Email")
        )
        dispatch_export["priority_score"] = (dispatch_export["net_recoverable_value"] / 100).round(1)

        csv_payload = dispatch_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Download Dispatch Payload for {target_action} (CSV)",
            data=csv_payload,
            file_name=f"dispatch_queue_{target_action.replace(' ', '_').lower()}.csv",
            mime="text/csv",
        )


# ==============================================================================
# 3. ANALYTICS & BEHAVIORAL EDA
# ==============================================================================
elif navigation_page == "Analytics & Behavioral EDA":
    st.title("📈 Behavioral & Exploratory Data Analytics")
    st.markdown(
        "Empirical investigation of customer usage patterns, support ticket loads, "
        "payment friction, and correlation drivers behind customer churn."
    )

    tab1, tab2, tab3 = st.tabs(["Churn Drivers & Friction", "Usage & Engagement Patterns", "Correlation Heatmap"])

    with tab1:
        row1_c1, row1_c2 = st.columns(2)
        with row1_c1:
            st.plotly_chart(fig_churn_distribution(raw_df), use_container_width=True)
        with row1_c2:
            st.plotly_chart(fig_plan_churn(raw_df), use_container_width=True)

        row2_c1, row2_c2 = st.columns(2)
        with row2_c1:
            st.plotly_chart(fig_payment_failures_vs_churn(raw_df), use_container_width=True)
        with row2_c2:
            st.plotly_chart(fig_support_vs_churn(raw_df), use_container_width=True)

    with tab2:
        col2_a, col2_b = st.columns(2)
        with col2_a:
            st.plotly_chart(fig_usage_vs_churn(raw_df), use_container_width=True)
        with col2_b:
            st.plotly_chart(fig_last_login_vs_churn(raw_df), use_container_width=True)

        col2_c, col2_d = st.columns(2)
        with col2_c:
            st.plotly_chart(fig_tenure_vs_churn(raw_df), use_container_width=True)
        with col2_d:
            st.plotly_chart(fig_high_risk_groups(raw_df), use_container_width=True)

    with tab3:
        st.subheader("Correlation Heatmap of Behavioral Signals")
        st.plotly_chart(fig_correlation(raw_df), use_container_width=True)
        st.caption(
            "Observation: Payment failures, login dormancy, and support ticket frequency show strong positive "
            "correlations with churn. Weekly active usage shows significant negative correlation."
        )


# ==============================================================================
# 4. CUSTOMER DEEP-DIVE & AI OUTREACH
# ==============================================================================
elif navigation_page == "Customer Deep-Dive & AI Outreach":
    st.title("🔍 Individual Customer Diagnostic & AI Outreach Engine")
    st.markdown(
        "Diagnose any customer's risk profile, view **local SHAP feature attribution**, "
        "and generate tailored, empathetic recovery messaging powered by AI."
    )

    selected_user_id = st.selectbox(
        "Search or Select Customer ID:",
        options=scored_df["user_id"].tolist(),
        index=0,
    )

    customer = scored_df[scored_df["user_id"] == selected_user_id].iloc[0]

    col_profile, col_diagnostic = st.columns([1, 1.4])

    with col_profile:
        st.subheader("👤 Account Profile & Unit Economics")
        p_c1, p_c2 = st.columns(2)
        p_c1.write(f"**Customer ID:** `{customer['user_id']}`")
        p_c1.write(f"**Plan Tier:** `{customer['plan_type']}`")
        p_c1.write(f"**Monthly Fee:** `₹{customer['monthly_fee']:.0f}`")
        p_c1.write(f"**Tenure:** `{customer['tenure_months']} months`")

        p_c2.write(f"**Weekly Usage:** `{customer['avg_weekly_usage_hours']:.1f} hrs`")
        p_c2.write(f"**Support Tickets:** `{customer['support_tickets']}`")
        p_c2.write(f"**Payment Failures:** `{customer['payment_failures']}`")
        p_c2.write(f"**Last Login:** `{customer['last_login_days_ago']} days ago`")

        st.markdown("---")
        st.subheader("💼 Financial Scorecard")
        f_c1, f_c2 = st.columns(2)
        f_c1.metric("Est. Remaining LTV", f"₹{customer.get('expected_remaining_ltv', 0):,.2f}")
        f_c1.metric("Intervention Cost", f"₹{customer.get('intervention_cost', 0):,.2f}")
        f_c2.metric("Gross Revenue at Risk", f"₹{customer['revenue_at_risk']:,.2f}")
        f_c2.metric("Net Recoverable Margin", f"₹{customer.get('net_recoverable_value', 0):,.2f}", f"{customer.get('roi_multiple', 0):.1f}x ROI")

    with col_diagnostic:
        st.subheader("🎯 Retention Diagnostics & Strategy")
        d1, d2 = st.columns(2)
        d1.metric("Churn Probability", f"{customer['churn_probability']:.1%}")
        d1.metric("Risk Tier", f"{customer['risk_level']}")
        d2.metric("Recovery Probability", f"{customer['recovery_probability']:.1%}")
        d2.info(f"**Action:** {customer['recommended_action']}")

        # Local SHAP Explanation Chart
        st.write("### 🔬 Local SHAP Feature Attribution")
        st.caption("How specific behavioral features pushed this customer's churn risk up (Red) or down (Green):")
        shap_fig = get_local_shap_explanation(model, preprocessor, customer, max_features=7)
        st.plotly_chart(shap_fig, use_container_width=True)
        st.caption(explain_customer_risk(customer))

    st.markdown("---")
    st.subheader("✉️ AI-Powered Personalized Retention Outreach")

    col_ai1, col_ai2 = st.columns([1, 1.2])

    with col_ai1:
        custom_instructions = st.text_input(
            "Optional Agent Guidance (e.g. 'Offer 15% discount', 'Focus on billing ease'):",
            "",
        )
        api_key_override = st.text_input(
            "OpenAI API Key (Optional — reads from .env if omitted):",
            type="password",
            help="If no API key is provided, the system utilizes a high-converting contextual fallback template.",
        )
        generate_btn = st.button("✨ Generate Personalized Recovery Message", type="primary")

    with col_ai2:
        if generate_btn or "current_msg" in st.session_state:
            if generate_btn:
                with st.spinner("Synthesizing personalized outreach message..."):
                    generated_msg = generate_llm_recovery_message(
                        customer,
                        api_key=api_key_override if api_key_override else None,
                        custom_instructions=custom_instructions,
                    )
                    st.session_state["current_msg"] = generated_msg

            st.text_area(
                "Generated Message Preview (Ready to Send)",
                value=st.session_state.get("current_msg", ""),
                height=240,
            )
        else:
            st.info("Click 'Generate Personalized Recovery Message' to generate outreach copy.")


# ==============================================================================
# 5. REVENUE RECOVERY SIMULATOR & DECISION FRONTIER
# ==============================================================================
elif navigation_page == "Revenue Recovery Simulator":
    st.title("🧮 Interactive Revenue Recovery Simulator")
    st.markdown(
        "Simulate financial outcomes and discover the **Optimal Cost-Sensitive Decision Frontier** "
        "that maximizes net profit."
    )

    sim_tab1, sim_tab2 = st.tabs(["Strategic Scenario Simulator", "Cost-Sensitive Decision Frontier"])

    with sim_tab1:
        col_sim_in1, col_sim_in2 = st.columns(2)

        with col_sim_in1:
            st.subheader("Cohort Parameters")
            sim_cohort = st.number_input(
                "Customer Cohort Size",
                min_value=100,
                max_value=100000,
                value=len(scored_df),
                step=100,
            )
            sim_fee = st.slider(
                "Average Monthly Fee per Account (₹)",
                min_value=50.0,
                max_value=1500.0,
                value=float(scored_df["monthly_fee"].mean()),
                step=10.0,
            )
            sim_churn_rate = st.slider(
                "Baseline Projected Churn Rate (%)",
                min_value=5.0,
                max_value=85.0,
                value=float((scored_df["churn_probability"] >= 0.50).mean() * 100),
                step=1.0,
            )

        with col_sim_in2:
            st.subheader("Intervention Targets")
            sim_horizon = st.slider(
                "Evaluation Forecast Horizon (Months)",
                min_value=1,
                max_value=24,
                value=6,
                step=1,
            )
            sim_recovery_rate = st.slider(
                "Target Strategy Recovery Success Rate (%)",
                min_value=5.0,
                max_value=80.0,
                value=35.0,
                step=1.0,
            )

        # Simulation calculations
        sim_at_risk_cohort = sim_cohort * (sim_churn_rate / 100.0)
        sim_gross_risk = sim_at_risk_cohort * sim_fee * sim_horizon
        sim_recovered_rev = sim_gross_risk * (sim_recovery_rate / 100.0)
        sim_unrecovered_risk = sim_gross_risk - sim_recovered_rev
        sim_cost = sim_at_risk_cohort * 35.0  # Average blended intervention cost
        sim_net_margin = sim_recovered_rev - sim_cost
        sim_recovered_customers = sim_at_risk_cohort * (sim_recovery_rate / 100.0)

        st.markdown("---")
        st.subheader("Simulation Projections")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Projected Churn Customers", f"{int(sim_at_risk_cohort):,} accounts")
        m2.metric("Gross Revenue at Risk", f"₹{sim_gross_risk:,.2f}")
        m3.metric("Net Recovered Profit", f"₹{sim_net_margin:,.2f}", f"{(sim_net_margin / max(sim_cost, 1)):.1f}x ROI")
        m4.metric("Saved Customers", f"{int(sim_recovered_customers):,} retained accounts")

        # Waterfall comparison chart
        sim_chart_df = pd.DataFrame({
            "Financial Metric": ["Gross Revenue at Risk", "Intervention Cost", "Net Recovered Profit", "Net Unrecovered Risk"],
            "Amount (₹)": [sim_gross_risk, sim_cost, sim_net_margin, sim_unrecovered_risk],
        })
        fig_sim = px.bar(
            sim_chart_df,
            x="Financial Metric",
            y="Amount (₹)",
            color="Financial Metric",
            color_discrete_map={
                "Gross Revenue at Risk": "#e74c3c",
                "Intervention Cost": "#e67e22",
                "Net Recovered Profit": "#2ecc71",
                "Net Unrecovered Risk": "#95a5a6",
            },
            text_auto=",.0f",
        )
        fig_sim.update_layout(showlegend=False)
        st.plotly_chart(fig_sim, use_container_width=True)

    with sim_tab2:
        st.subheader("Cost-Sensitive Decision Threshold Frontier")
        st.markdown(
            "In commercial data science, standard 0.50 classification thresholds are rarely profit-optimal. "
            "By varying the threshold, we balance the cost of outreach against saved subscription revenue."
        )

        threshold_sweep = np.linspace(0.15, 0.85, 29)
        frontier_rows = []

        total_saved_revenue = []
        total_costs = []
        net_profits = []

        for thresh in threshold_sweep:
            target_mask = scored_df["churn_probability"] >= thresh
            sub = scored_df[target_mask]
            rec_rev = sub["estimated_recoverable_revenue"].sum()
            cost = sub["intervention_cost"].sum()
            net = rec_rev - cost
            frontier_rows.append({
                "Threshold": round(thresh, 2),
                "Recovered Revenue": rec_rev,
                "Intervention Cost": cost,
                "Net Profit": net,
                "Customers Targeted": len(sub),
            })

        frontier_df = pd.DataFrame(frontier_rows)
        best_row = frontier_df.loc[frontier_df["Net Profit"].idxmax()]

        st.metric(
            "Optimal Churn Decision Cutoff",
            f"P(Churn) >= {best_row['Threshold']:.2f}",
            f"Yields Maximum Net Profit of ₹{best_row['Net Profit']:,.2f} across {int(best_row['Customers Targeted'])} accounts"
        )

        fig_frontier = px.line(
            frontier_df,
            x="Threshold",
            y=["Recovered Revenue", "Intervention Cost", "Net Profit"],
            labels={"value": "Amount in INR (₹)", "Threshold": "Classification Probability Cutoff P(Churn)"},
            title="Net Profit Frontier vs Churn Probability Decision Threshold",
            color_discrete_map={
                "Recovered Revenue": "#3498db",
                "Intervention Cost": "#e74c3c",
                "Net Profit": "#2ecc71",
            },
        )
        fig_frontier.add_vline(x=best_row["Threshold"], line_dash="dash", line_color="#27ae60", annotation_text="Profit Optimum")
        st.plotly_chart(fig_frontier, use_container_width=True)


# ==============================================================================
# 6. MODEL GOVERNANCE & EXPLAINABILITY
# ==============================================================================
elif navigation_page == "Model Governance & Explainability":
    st.title("🛡️ Model Governance, Evaluation & Explainability")
    st.markdown(
        "Rigorous benchmarking of classification algorithms, feature importance rankings, "
        "and business rationale for the Recall & ROC-AUC optimization framework."
    )

    st.subheader("1. Multi-Model Benchmark Comparison")
    if METRICS_PATH.exists():
        metrics_df = pd.read_csv(METRICS_PATH)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    else:
        st.info("Run model training to populate metrics table.")

    st.markdown("---")

    col_gov1, col_gov2 = st.columns(2)

    with col_gov1:
        st.subheader("2. Global Feature Importances")
        imp_df = get_feature_importances(model, preprocessor)
        if imp_df is not None:
            fig_imp = px.bar(
                imp_df.head(10),
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Blues",
                title="Top 10 Drivers of Customer Churn",
            )
            fig_imp.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance display unavailable for current model architecture.")

    with col_gov2:
        st.subheader("3. Business Optimization Rationale")
        st.markdown(
            """
            #### Why Prioritize Recall & ROC-AUC Over Accuracy?
            
            1. **Asymmetric Cost of Errors:**
               - **False Positive (Type I Error):** The model flags a safe customer as churning. Cost: Sending a courteous retention email or survey (negligible).
               - **False Negative (Type II Error):** The model fails to identify a customer about to churn. Cost: Permanent loss of recurring subscription revenue (₹1,194 – ₹5,592 per account).
               
            2. **Revenue-Optimal Selection Metric:**
               $$\\text{Score} = 0.45 \\times \\text{Recall} + 0.40 \\times \\text{ROC-AUC} + 0.15 \\times \\text{F1}$$
               
            3. **Unit Economics & Profitable Retention:**
               Interventions with negative Net Expected Value (Cost > Recovered Revenue) are flagged so retention managers preserve budget.
               
            4. **Ethical AI & Compliance:**
               Customer-level risk explanations utilize non-causal language (*"contributing factors"*) rather than asserting direct causality.
            """
        )

