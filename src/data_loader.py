"""src/data_loader.py – Data loading and preprocessing utilities."""
import numpy as np
import pandas as pd
import streamlit as st


CHURN_ALIASES = {"churn", "churned", "attrition", "ischurned", "is_churned", "target"}
CATEGORY_COLS = [
    "orders_Electronics", "orders_Fashion", "orders_Home_Kitchen",
    "orders_Beauty", "orders_Books", "orders_Sports", "orders_Toys",
]


@st.cache_data(show_spinner="Loading dataset…")
def load_data(source) -> pd.DataFrame:
    """Load CSV from an uploaded file object or a file path string."""
    if hasattr(source, "read"):
        df = pd.read_csv(source)
    else:
        df = pd.read_csv(source)

    # Normalise blank / sentinel strings to NaN
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].replace(["", "NA", "N/A", "null", "None", "nan"], np.nan)

    return df


def detect_churn_column(df: pd.DataFrame) -> str | None:
    """Return the first churn-like column name, or None."""
    for col in df.columns:
        if col.lower().replace("_", "").replace(" ", "") in CHURN_ALIASES:
            return col
    # Fallback: first binary column
    for col in df.columns:
        if df[col].nunique(dropna=True) == 2:
            return col
    return None


def get_summary(df: pd.DataFrame) -> dict:
    """Return basic dataset statistics."""
    return {
        "rows": len(df),
        "cols": df.shape[1],
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "churn_rate": df["Churn"].mean() if "Churn" in df.columns else None,
    }


def get_category_cols(df: pd.DataFrame) -> list[str]:
    """Return category purchase columns that actually exist in df."""
    return [c for c in CATEGORY_COLS if c in df.columns]
