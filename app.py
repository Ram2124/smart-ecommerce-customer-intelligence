"""
app.py – Smart E-Commerce Customer Intelligence Dashboard
=========================================================
A production-quality Streamlit application showcasing:
  • Exploratory Data Analysis
  • K-Means Customer Segmentation (with elbow method)
  • Multi-model Churn Prediction (Logistic Regression, Random Forest, Gradient Boosting)
  • Collaborative-Filtering Product Recommendation Engine
  • Interactive Data Explorer with download

Author : [Your Name]
Dataset : Synthetic e-commerce customer data (5 000 records)
"""

import hashlib
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

# ── make sure src/ is importable ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import (
    load_data, detect_churn_column, get_summary, get_category_cols
)
from src.models import run_kmeans, prepare_churn_target, train_all_classifiers
from src.recommender import build_similarity_matrix, recommend_for_customer, recommend_for_segment
from src.ui_components import (
    kpi_row, distribution_chart, category_bar, scatter_plot,
    correlation_heatmap, segment_scatter, elbow_chart,
    confusion_matrix_chart, model_comparison_chart, recommendation_cards,
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart E-Commerce Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  section[data-testid="stSidebar"] { background-color: #1A1D27; }
  .block-container { padding-top: 1.5rem; }
  div[data-testid="metric-container"] {
    background: #1A1D27; border-radius: 10px; padding: 12px 16px;
  }
  h1 { color: #6C63FF; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛒 E-Commerce\n### Customer Intelligence")
    st.divider()
    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

    DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "data", "ecommerce_customers.csv")
    use_default  = os.path.isfile(DEFAULT_PATH)

    if uploaded is None and not use_default:
        st.info("No default dataset found.\nRun `python generate_dataset.py` first,\nor upload your own CSV.")

    st.divider()
    st.markdown("**Navigation**")
    st.markdown("Use the tabs above to explore each module.")
    st.divider()
    st.caption("Built with Streamlit · scikit-learn · Plotly")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
if uploaded is not None:
    df_raw = load_data(uploaded)
elif use_default:
    df_raw = load_data(DEFAULT_PATH)
    st.sidebar.success(f"Using default dataset: **ecommerce_customers.csv**")
else:
    st.title("🛒 Smart E-Commerce Customer Intelligence")
    st.warning("Please upload a CSV file or run `python generate_dataset.py` to create the sample dataset.")
    st.stop()

df = df_raw.copy()
stats = get_summary(df)
st.sidebar.success(f"✅  {stats['rows']:,} rows × {stats['cols']} cols loaded")

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("🛒 Smart E-Commerce Customer Intelligence")
st.caption("Customer analytics · segmentation · churn prediction · product recommendations")
kpi_row(stats)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "👥 Segmentation",
    "⚠️ Churn Prediction",
    "🛍️ Recommendations",
    "🔎 Data Explorer",
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 – OVERVIEW / EDA
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("📊 Exploratory Data Analysis")

    numeric_cols    = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    col_l, col_r = st.columns(2)

    with col_l:
        if numeric_cols:
            sel_num = st.selectbox("Numerical feature", numeric_cols, key="eda_num")
            st.plotly_chart(distribution_chart(df, sel_num), use_container_width=True)

    with col_r:
        if categorical_cols:
            sel_cat = st.selectbox("Categorical feature", categorical_cols, key="eda_cat")
            st.plotly_chart(category_bar(df, sel_cat), use_container_width=True)

    # Scatter plot
    if len(numeric_cols) >= 2:
        st.markdown("#### Scatter Explorer")
        sc1, sc2, sc3 = st.columns(3)
        x_feat   = sc1.selectbox("X axis",     numeric_cols, index=0, key="sc_x")
        y_feat   = sc2.selectbox("Y axis",     numeric_cols, index=min(1, len(numeric_cols)-1), key="sc_y")
        col_feat = sc3.selectbox("Colour by",  ["None"] + categorical_cols, key="sc_col")
        color_arg = None if col_feat == "None" else col_feat
        st.plotly_chart(scatter_plot(df, x_feat, y_feat, color_arg), use_container_width=True)

    # Correlation heatmap
    st.markdown("#### Correlation Heatmap")
    hm = correlation_heatmap(df)
    if hm:
        st.plotly_chart(hm, use_container_width=True)
    else:
        st.info("Need at least 2 numerical columns for a heatmap.")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 – SEGMENTATION
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("👥 Customer Segmentation — K-Means Clustering")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    usable = [c for c in numeric_cols if df[c].nunique(dropna=True) > 1]

    if len(usable) < 2:
        st.warning("At least 2 numerical features are required for segmentation.")
    else:
        seg_l, seg_r = st.columns([2, 1])
        with seg_l:
            features = st.multiselect(
                "Select features for clustering",
                usable,
                default=[c for c in ["Tenure_Months", "Total_Spend", "Num_Orders",
                                      "Days_Since_Last_Purchase", "Satisfaction_Score"]
                          if c in usable][:5] or usable[:4],
            )
        with seg_r:
            k = st.slider("Number of segments (k)", 2, 8, 4)

        if len(features) < 2:
            st.info("Select at least 2 features.")
        else:
            with st.spinner("Running K-Means…"):
                labels, inertias = run_kmeans(df, features, k)

            segmented = df.copy()
            segmented["Customer_Segment"] = labels + 1

            st.success(f"✅  Created **{k} customer segments** from {len(segmented):,} records.")

            c_left, c_right = st.columns(2)
            with c_left:
                seg_counts = (
                    segmented["Customer_Segment"]
                    .value_counts().sort_index()
                    .rename("Customers").rename_axis("Segment")
                    .reset_index()
                )
                seg_counts["Segment"] = "Segment " + seg_counts["Segment"].astype(str)
                import plotly.express as px
                fig_pie = px.pie(seg_counts, names="Segment", values="Customers",
                                  title="Segment Distribution", color_discrete_sequence=px.colors.qualitative.Vivid)
                st.plotly_chart(fig_pie, use_container_width=True)

            with c_right:
                st.plotly_chart(elbow_chart(inertias), use_container_width=True)

            # Segment scatter
            xf = st.selectbox("Scatter X", features, key="seg_x")
            yf = st.selectbox("Scatter Y", [f for f in features if f != xf], key="seg_y")
            st.plotly_chart(segment_scatter(segmented, xf, yf), use_container_width=True)

            # Segment summary table
            st.markdown("#### Segment Profiles")
            num_feats = [c for c in features if c in segmented.select_dtypes(include=np.number).columns]
            profile = segmented.groupby("Customer_Segment")[num_feats].mean().round(1)
            profile.index = ["Segment " + str(i) for i in profile.index]
            st.dataframe(profile, use_container_width=True)

            st.download_button(
                "⬇️ Download segmented customers",
                segmented.to_csv(index=False).encode(),
                "segmented_customers.csv",
                "text/csv",
            )

            # Store for Recommendations tab
            st.session_state["segmented_df"] = segmented

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 – CHURN PREDICTION
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("⚠️ Churn Prediction — Multi-Model Comparison")

    churn_col = detect_churn_column(df)

    if churn_col is None:
        st.info("No binary churn column detected. Upload a dataset with a 'Churn' column.")
    else:
        st.markdown(f"**Target column detected:** `{churn_col}`")

        target = prepare_churn_target(df[churn_col])
        valid  = target.notna()
        X_full = df.loc[valid].drop(columns=[churn_col, "CustomerID"], errors="ignore")
        y_full = target.loc[valid].astype(int)

        if y_full.nunique() != 2:
            st.warning("Target column must have exactly 2 distinct values after conversion.")
        else:
            from sklearn.model_selection import train_test_split
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
            )

            df_hash = hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()

            with st.spinner("Training Logistic Regression, Random Forest & Gradient Boosting…"):
                results = train_all_classifiers(
                    df_hash,
                    X_tr.to_json(), y_tr.to_json(),
                    X_te.to_json(), y_te.to_json(),
                )

            # Model comparison chart
            st.plotly_chart(model_comparison_chart(results), use_container_width=True)

            # Per-model details
            best_model = max(results, key=lambda n: results[n]["auc"] or 0)
            st.markdown(f"🏆 **Best model by ROC-AUC: {best_model}**")

            sel_model = st.selectbox("Inspect model details", list(results.keys()))
            res = results[sel_model]

            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy", f"{res['accuracy']:.2%}")
            m2.metric("F1 Score", f"{res['f1']:.3f}")
            m3.metric("ROC-AUC",  f"{res['auc']:.3f}" if res["auc"] else "N/A")

            d_left, d_right = st.columns(2)
            with d_left:
                st.plotly_chart(confusion_matrix_chart(res["cm"]), use_container_width=True)
            with d_right:
                st.markdown("**Classification Report**")
                st.code(res["report"], language="text")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 – RECOMMENDATIONS
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("🛍️ Product Recommendation Engine")
    st.caption("Item-based collaborative filtering using cosine similarity on purchase history.")

    cat_cols = get_category_cols(df)

    if not cat_cols:
        st.warning("No category purchase columns found. The recommendation engine needs `orders_*` columns.")
    else:
        with st.spinner("Building similarity matrix…"):
            sim_matrix, avail_cats = build_similarity_matrix(df)

        rec_tabs = st.tabs(["👤 Customer-Level", "👥 Segment-Level"])

        # ── Customer-level ────────────────────────────────────────────────────
        with rec_tabs[0]:
            if "CustomerID" not in df.columns:
                st.info("CustomerID column is required for customer-level recommendations.")
            else:
                cid = st.selectbox("Select a customer", df["CustomerID"].head(200).tolist())
                n_recs = st.slider("Number of category recommendations", 2, 5, 3)

                recs = recommend_for_customer(df, sim_matrix, avail_cats, cid, n_recs)

                # Show customer profile
                cust_row = df[df["CustomerID"] == cid].iloc[0]
                st.markdown("#### Customer Profile")
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Total Spend",    f"₹{cust_row.get('Total_Spend', 'N/A'):,}" if pd.notna(cust_row.get("Total_Spend")) else "N/A")
                p2.metric("Orders",         str(cust_row.get("Num_Orders", "N/A")))
                p3.metric("Membership",     str(cust_row.get("Membership_Tier", "N/A")))
                p4.metric("Pref. Category", str(cust_row.get("Preferred_Category", "N/A")))

                st.markdown("#### 🎯 Recommended Categories & Products")
                recommendation_cards(recs)

        # ── Segment-level ─────────────────────────────────────────────────────
        with rec_tabs[1]:
            seg_df = st.session_state.get("segmented_df")
            if seg_df is None:
                st.info("Run segmentation first (👥 Segmentation tab) to see segment-level recommendations.")
            else:
                seg_ids = sorted(seg_df["Customer_Segment"].unique())
                sel_seg = st.selectbox("Select segment", [f"Segment {s}" for s in seg_ids])
                seg_id  = int(sel_seg.split()[-1])

                recs_seg = recommend_for_segment(seg_df, avail_cats, seg_id)
                st.markdown(f"#### 🎯 Top categories for **{sel_seg}**")
                recommendation_cards(recs_seg[:3])

                # Category purchase breakdown chart
                import plotly.express as px
                cat_avg = seg_df[seg_df["Customer_Segment"] == seg_id][avail_cats].mean().reset_index()
                cat_avg.columns = ["Category", "Avg Orders"]
                cat_avg["Category"] = cat_avg["Category"].str.replace("orders_", "").str.replace("_", " ")
                fig = px.bar(cat_avg.sort_values("Avg Orders", ascending=True),
                              x="Avg Orders", y="Category", orientation="h",
                              title=f"Avg Category Orders — {sel_seg}",
                              color="Avg Orders", color_continuous_scale="Viridis")
                st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 – DATA EXPLORER
# ──────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("🔎 Customer Data Explorer")

    # Filters
    with st.expander("🔽 Filters", expanded=True):
        f1, f2, f3 = st.columns(3)
        if "Membership_Tier" in df.columns:
            tiers = ["All"] + sorted(df["Membership_Tier"].dropna().unique().tolist())
            sel_tier = f1.selectbox("Membership Tier", tiers)
        else:
            sel_tier = "All"

        if "Location" in df.columns:
            locs = ["All"] + sorted(df["Location"].dropna().unique().tolist())
            sel_loc = f2.selectbox("Location", locs)
        else:
            sel_loc = "All"

        if "Churn" in df.columns:
            churn_filter = f3.selectbox("Churn Status", ["All", "Churned (1)", "Active (0)"])
        else:
            churn_filter = "All"

    filtered = df.copy()
    if sel_tier != "All":
        filtered = filtered[filtered["Membership_Tier"] == sel_tier]
    if sel_loc != "All":
        filtered = filtered[filtered["Location"] == sel_loc]
    if churn_filter == "Churned (1)":
        filtered = filtered[filtered["Churn"] == 1]
    elif churn_filter == "Active (0)":
        filtered = filtered[filtered["Churn"] == 0]

    st.caption(f"Showing {len(filtered):,} of {len(df):,} records")
    st.dataframe(filtered, use_container_width=True, height=460)

    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "⬇️ Download filtered data (CSV)",
        filtered.to_csv(index=False).encode(),
        "filtered_customers.csv", "text/csv",
    )
    buf = filtered.describe().T.reset_index()
    dl2.download_button(
        "⬇️ Download summary statistics (CSV)",
        buf.to_csv(index=False).encode(),
        "summary_stats.csv", "text/csv",
    )
