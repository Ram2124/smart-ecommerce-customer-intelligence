import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Smart E-Commerce Intelligence", page_icon="🛒", layout="wide")

st.title("🛒 Smart E-Commerce Customer Intelligence")
st.caption("Customer analytics • segmentation • churn prediction • interactive business insights")

@st.cache_data

def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

uploaded = st.sidebar.file_uploader("Upload ecommerce customer CSV", type=["csv"])

if uploaded is None:
    st.info("Upload your ecommerce customer CSV from the sidebar to start the analysis.")
    st.markdown("### Expected data")
    st.write("The dashboard works with mixed numerical and categorical customer data and can automatically detect common churn columns such as `churn`, `churned`, or `attrition`.")
    st.stop()

df = load_data(uploaded).copy()

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].replace(["", "NA", "N/A", "null", "None"], np.nan)

st.sidebar.success(f"Loaded {len(df):,} rows × {df.shape[1]} columns")

missing = int(df.isna().sum().sum())
duplicates = int(df.duplicated().sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers / Records", f"{len(df):,}")
c2.metric("Features", df.shape[1])
c3.metric("Missing Values", f"{missing:,}")
c4.metric("Duplicate Rows", f"{duplicates:,}")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "👥 Segmentation", "⚠️ Churn Prediction", "🔎 Customer Data"])

with tab1:
    st.subheader("Exploratory Data Analysis")
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    categorical = df.select_dtypes(exclude=np.number).columns.tolist()

    if numeric:
        selected_num = st.selectbox("Numerical feature", numeric)
        fig = px.histogram(df, x=selected_num, marginal="box", title=f"Distribution of {selected_num}")
        st.plotly_chart(fig, use_container_width=True)

    if categorical:
        selected_cat = st.selectbox("Categorical feature", categorical)
        counts = df[selected_cat].fillna("Missing").astype(str).value_counts().head(15).reset_index()
        counts.columns = [selected_cat, "Count"]
        fig = px.bar(counts, x=selected_cat, y="Count", title=f"Top values in {selected_cat}")
        st.plotly_chart(fig, use_container_width=True)

    if len(numeric) >= 2:
        xcol, ycol = st.columns(2)
        x_feature = xcol.selectbox("X feature", numeric, index=0)
        y_feature = ycol.selectbox("Y feature", numeric, index=1)
        fig = px.scatter(df, x=x_feature, y=y_feature, title=f"{x_feature} vs {y_feature}")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Customer Segmentation — K-Means")
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    usable = [c for c in numeric if df[c].nunique(dropna=True) > 1]

    if len(usable) < 2:
        st.warning("At least two numerical features are required for K-Means segmentation.")
    else:
        features = st.multiselect("Select customer features", usable, default=usable[:min(4, len(usable))])
        k = st.slider("Number of customer segments", 2, 8, 4)
        if len(features) >= 2:
            X = df[features].apply(pd.to_numeric, errors="coerce")
            X = SimpleImputer(strategy="median").fit_transform(X)
            X = StandardScaler().fit_transform(X)
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(X)
            segmented = df.copy()
            segmented["Customer_Segment"] = labels + 1
            st.success(f"Created {k} customer segments from {len(segmented):,} records.")
            a, b = st.columns(2)
            with a:
                st.dataframe(segmented["Customer_Segment"].value_counts().sort_index().rename("Customers"), use_container_width=True)
            with b:
                fig = px.scatter(segmented, x=features[0], y=features[1], color="Customer_Segment", title="Customer Segments")
                st.plotly_chart(fig, use_container_width=True)
            st.download_button("Download segmented customers", segmented.to_csv(index=False).encode(), "segmented_customers.csv", "text/csv")

with tab3:
    st.subheader("Customer Churn Prediction")
    churn_candidates = [c for c in df.columns if c.lower().replace("_", "").replace(" ", "") in {"churn", "churned", "attrition", "is churned", "ischurned".replace(" ", "")}]
    churn_col = churn_candidates[0] if churn_candidates else None
    if churn_col is None:
        possible = [c for c in df.columns if df[c].nunique(dropna=True) == 2]
        if possible:
            churn_col = st.selectbox("Select the binary churn/target column", possible)

    if churn_col is None:
        st.info("No binary churn column was detected. Add or select a churn target to train the model.")
    else:
        target = df[churn_col].copy()
        if target.dtype == object:
            target = target.astype(str).str.strip().str.lower().map({"yes": 1, "y": 1, "true": 1, "1": 1, "churn": 1, "no": 0, "n": 0, "false": 0, "0": 0, "active": 0})
        else:
            target = pd.to_numeric(target, errors="coerce")
        valid = target.notna()
        work = df.loc[valid].drop(columns=[churn_col]).copy()
        target = target.loc[valid].astype(int)
        if target.nunique() != 2:
            st.warning("The selected target must contain exactly two classes after conversion.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(work, target, test_size=0.2, random_state=42, stratify=target)
            nums = X_train.select_dtypes(include=np.number).columns.tolist()
            cats = X_train.select_dtypes(exclude=np.number).columns.tolist()
            transformers = []
            if nums:
                transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), nums))
            if cats:
                transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cats))
            pre = ColumnTransformer(transformers=transformers)
            clf = Pipeline([("preprocess", pre), ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))])
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            acc = accuracy_score(y_test, pred)
            st.metric("Test Accuracy", f"{acc:.2%}")
            cm = confusion_matrix(y_test, pred)
            st.write("Confusion Matrix")
            st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))
            st.text(classification_report(y_test, pred, zero_division=0))

with tab4:
    st.subheader("Customer Data")
    st.dataframe(df, use_container_width=True, height=500)
    st.download_button("Download cleaned view", df.to_csv(index=False).encode(), "ecommerce_customer_data.csv", "text/csv")
