"""src/recommender.py – Item-based collaborative filtering recommendation engine."""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

CATEGORY_COLS = [
    "orders_Electronics", "orders_Fashion", "orders_Home_Kitchen",
    "orders_Beauty", "orders_Books", "orders_Sports", "orders_Toys",
]

CATEGORY_DISPLAY = {
    "orders_Electronics":   "📱 Electronics",
    "orders_Fashion":       "👗 Fashion",
    "orders_Home_Kitchen":  "🏠 Home & Kitchen",
    "orders_Beauty":        "💄 Beauty",
    "orders_Books":         "📚 Books",
    "orders_Sports":        "🏋️ Sports",
    "orders_Toys":          "🧸 Toys",
}

CATEGORY_PRODUCTS = {
    "orders_Electronics":  ["Wireless Earbuds", "Smart Watch", "Laptop Stand", "Bluetooth Speaker", "USB-C Hub"],
    "orders_Fashion":      ["Running Shoes", "Casual T-Shirt", "Denim Jacket", "Sunglasses", "Leather Wallet"],
    "orders_Home_Kitchen": ["Air Fryer", "Coffee Maker", "Non-Stick Pan Set", "Robot Vacuum", "Knife Set"],
    "orders_Beauty":       ["Face Serum", "Moisturiser SPF50", "Hair Mask", "Perfume Set", "Lip Kit"],
    "orders_Books":        ["Data Science Handbook", "Python Crash Course", "Atomic Habits", "Deep Work", "The Almanack"],
    "orders_Sports":       ["Yoga Mat", "Resistance Bands", "Protein Powder", "Gym Gloves", "Water Bottle"],
    "orders_Toys":         ["LEGO Set", "Board Game", "Puzzle 1000pc", "RC Car", "Craft Kit"],
}


@st.cache_data(show_spinner="Building recommendation engine…")
def build_similarity_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return (cosine similarity matrix, list of valid category columns)."""
    avail = [c for c in CATEGORY_COLS if c in df.columns]
    if not avail:
        return np.array([]), []
    matrix = df[avail].fillna(0).values.astype(float)
    # Customer-to-customer cosine similarity
    sim = cosine_similarity(matrix)
    return sim, avail


def recommend_for_customer(
    df: pd.DataFrame,
    sim_matrix: np.ndarray,
    cat_cols: list[str],
    customer_id: str,
    top_n: int = 3,
) -> list[dict]:
    """
    Return top_n category recommendations for a given customer.
    Uses weighted average of similar customers' purchase vectors.
    """
    if len(sim_matrix) == 0:
        return []

    idx = df.index[df["CustomerID"] == customer_id].tolist()
    if not idx:
        return []
    i = idx[0]

    sims = sim_matrix[i].copy()
    sims[i] = 0  # exclude self

    # Weighted purchase vector of neighbours
    weights = np.maximum(sims, 0)
    if weights.sum() == 0:
        return []

    weighted_purchases = (weights[:, None] * df[cat_cols].fillna(0).values).sum(axis=0)
    customer_purchases = df.loc[i, cat_cols].fillna(0).values

    # Score = neighbour affinity minus own purchase saturation
    scores = weighted_purchases / (customer_purchases + 1)
    top_idx = np.argsort(scores)[::-1][:top_n]

    recommendations = []
    for ci in top_idx:
        col = cat_cols[ci]
        display = CATEGORY_DISPLAY.get(col, col)
        products = CATEGORY_PRODUCTS.get(col, [])
        recommendations.append({
            "category": display,
            "score": float(scores[ci]),
            "products": products,
        })
    return recommendations


def recommend_for_segment(
    df: pd.DataFrame,
    cat_cols: list[str],
    segment_id: int,
) -> list[dict]:
    """Return top categories for a given cluster segment."""
    seg_df = df[df["Customer_Segment"] == segment_id]
    if seg_df.empty or not cat_cols:
        return []
    avg = seg_df[cat_cols].mean().sort_values(ascending=False)
    recs = []
    for col, score in avg.items():
        recs.append({
            "category": CATEGORY_DISPLAY.get(col, col),
            "avg_orders": round(score, 1),
            "products": CATEGORY_PRODUCTS.get(col, [])[:3],
        })
    return recs
