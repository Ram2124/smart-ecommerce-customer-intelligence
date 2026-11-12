"""src/ui_components.py – Reusable Streamlit UI helpers."""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st

PALETTE = px.colors.qualitative.Vivid


def kpi_row(stats: dict):
    """Render top KPI metric cards."""
    cols = st.columns(5)
    churn = stats.get("churn_rate")
    items = [
        ("👥 Customers",   f"{stats['rows']:,}",    None),
        ("📋 Features",    str(stats["cols"]),       None),
        ("❓ Missing",     f"{stats['missing']:,}",  "inverse" if stats["missing"] > 0 else "normal"),
        ("🔁 Duplicates",  f"{stats['duplicates']:,}", "inverse" if stats["duplicates"] > 0 else "normal"),
        ("⚠️ Churn Rate",  f"{churn:.1%}" if churn is not None else "N/A", "inverse" if churn and churn > 0.25 else "normal"),
    ]
    for col, (label, value, delta) in zip(cols, items):
        col.metric(label, value)


def distribution_chart(df: pd.DataFrame, col: str) -> go.Figure:
    fig = px.histogram(
        df, x=col, marginal="box",
        title=f"Distribution of {col}",
        color_discrete_sequence=[PALETTE[0]],
    )
    fig.update_layout(bargap=0.05)
    return fig


def category_bar(df: pd.DataFrame, col: str) -> go.Figure:
    counts = (
        df[col].fillna("Missing").astype(str)
        .value_counts().head(15).reset_index()
    )
    counts.columns = [col, "Count"]
    fig = px.bar(
        counts, x=col, y="Count",
        title=f"Top values in {col}",
        color=col, color_discrete_sequence=PALETTE,
    )
    return fig


def scatter_plot(df: pd.DataFrame, x: str, y: str, color_col: str | None = None) -> go.Figure:
    fig = px.scatter(
        df, x=x, y=y, color=color_col,
        title=f"{x} vs {y}",
        opacity=0.65,
        color_discrete_sequence=PALETTE,
        color_continuous_scale="Viridis",
    )
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] < 2:
        return None
    corr = num_df.corr()
    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        title="Feature Correlation Heatmap",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
    )
    return fig


def segment_scatter(segmented: pd.DataFrame, x: str, y: str) -> go.Figure:
    segmented = segmented.copy()
    segmented["Segment"] = "Segment " + segmented["Customer_Segment"].astype(str)
    fig = px.scatter(
        segmented, x=x, y=y, color="Segment",
        title=f"Customer Segments — {x} vs {y}",
        opacity=0.7,
        color_discrete_sequence=PALETTE,
    )
    return fig


def elbow_chart(inertias: dict) -> go.Figure:
    ks = list(inertias.keys())
    vals = list(inertias.values())
    fig = go.Figure(go.Scatter(x=ks, y=vals, mode="lines+markers",
                               marker=dict(size=8, color=PALETTE[1]),
                               line=dict(color=PALETTE[1])))
    fig.update_layout(title="Elbow Method – Optimal k", xaxis_title="k", yaxis_title="Inertia")
    return fig


def confusion_matrix_chart(cm: np.ndarray) -> go.Figure:
    labels = ["Not Churn", "Churn"]
    fig = px.imshow(
        cm, text_auto=True, x=labels, y=labels,
        color_continuous_scale="Blues",
        title="Confusion Matrix",
        labels=dict(x="Predicted", y="Actual"),
    )
    return fig


def model_comparison_chart(results: dict) -> go.Figure:
    names   = list(results.keys())
    acc     = [results[n]["accuracy"] for n in names]
    f1      = [results[n]["f1"]       for n in names]
    auc     = [results[n]["auc"] or 0 for n in names]
    fig = go.Figure()
    fig.add_bar(name="Accuracy", x=names, y=acc,  marker_color=PALETTE[0])
    fig.add_bar(name="F1 Score", x=names, y=f1,   marker_color=PALETTE[2])
    fig.add_bar(name="ROC-AUC",  x=names, y=auc,  marker_color=PALETTE[4])
    fig.update_layout(barmode="group", title="Model Comparison", yaxis_range=[0, 1.05])
    return fig


def recommendation_cards(recs: list[dict]):
    """Render recommendation result cards."""
    if not recs:
        st.info("Not enough data to generate recommendations.")
        return
    cols = st.columns(len(recs))
    for col, rec in zip(cols, recs):
        with col:
            st.markdown(f"### {rec['category']}")
            if "products" in rec and rec["products"]:
                for p in rec["products"][:3]:
                    st.markdown(f"- {p}")
            if "avg_orders" in rec:
                st.caption(f"Avg orders: {rec['avg_orders']}")
