"""src/models.py – ML model training and evaluation helpers."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import streamlit as st


# ── Segmentation ──────────────────────────────────────────────────────────────

def run_kmeans(df: pd.DataFrame, features: list[str], k: int):
    """Fit K-Means and return (labels, inertia_list_for_elbow)."""
    X = df[features].apply(pd.to_numeric, errors="coerce")
    X = SimpleImputer(strategy="median").fit_transform(X)
    X = StandardScaler().fit_transform(X)

    # Elbow data (k=2..8)
    inertias = {}
    for ki in range(2, min(10, len(df))):
        m = KMeans(n_clusters=ki, random_state=42, n_init=10)
        m.fit(X)
        inertias[ki] = m.inertia_

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    return labels, inertias


# ── Churn Prediction ──────────────────────────────────────────────────────────

CHURN_LABEL_MAP = {
    "yes": 1, "y": 1, "true": 1, "1": 1, "churn": 1,
    "no": 0,  "n": 0, "false": 0, "0": 0, "active": 0,
}

CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Random Forest":        RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=150, random_state=42),
}


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    nums = X_train.select_dtypes(include=np.number).columns.tolist()
    cats = X_train.select_dtypes(exclude=np.number).columns.tolist()
    transformers = []
    if nums:
        transformers.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
        ]), nums))
    if cats:
        transformers.append(("cat", Pipeline([
            ("imp",    SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cats))
    return ColumnTransformer(transformers)


def prepare_churn_target(series: pd.Series) -> pd.Series:
    """Coerce a churn column to binary int (0/1)."""
    if series.dtype == object:
        return series.astype(str).str.strip().str.lower().map(CHURN_LABEL_MAP)
    return pd.to_numeric(series, errors="coerce")


@st.cache_resource(show_spinner="Training models…")
def train_all_classifiers(df_hash: str, X_train_json: str, y_train_json: str, X_test_json: str, y_test_json: str):
    """Train all classifiers and return results dict (cached by df_hash)."""
    X_train = pd.read_json(X_train_json)
    y_train = pd.read_json(y_train_json, typ="series")
    X_test  = pd.read_json(X_test_json)
    y_test  = pd.read_json(y_test_json, typ="series")

    pre = build_preprocessor(X_train)
    results = {}
    for name, clf in CLASSIFIERS.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_train, y_train)
        pred      = pipe.predict(X_test)
        pred_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
        results[name] = {
            "pipeline":  pipe,
            "accuracy":  accuracy_score(y_test, pred),
            "f1":        f1_score(y_test, pred, zero_division=0),
            "auc":       roc_auc_score(y_test, pred_prob) if pred_prob is not None else None,
            "cm":        confusion_matrix(y_test, pred),
            "report":    classification_report(y_test, pred, zero_division=0),
            "pred":      pred,
            "y_test":    y_test,
        }
    return results
