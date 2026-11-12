"""
generate_dataset.py  –  Run once to create data/ecommerce_customers.csv
Usage: python generate_dataset.py
"""
import os, numpy as np, pandas as pd

SEED, N = 42, 5000
rng = np.random.default_rng(SEED)

def wchoice(opts, w, n):
    w = np.array(w, dtype=float); return rng.choice(opts, size=n, p=w/w.sum())

customer_ids  = [f"CUST{str(i).zfill(5)}" for i in range(1, N+1)]
ages          = rng.integers(18, 71, N)
genders       = wchoice(["Male","Female","Other"], [47,50,3], N)
locations     = wchoice(["Mumbai","Delhi","Bangalore","Chennai","Hyderabad","Pune","Kolkata","Ahmedabad"], [18,16,15,11,10,10,10,10], N)
annual_income = rng.normal(600_000, 200_000, N).clip(150_000, 2_000_000).astype(int)
tenure_months = rng.integers(1, 73, N)
num_orders    = (tenure_months * rng.uniform(0.3, 1.5, N)).clip(1).astype(int)
avg_order_val = rng.normal(1800, 600, N).clip(300, 8000).astype(int)
total_spend   = (num_orders * avg_order_val * rng.uniform(0.8, 1.2, N)).astype(int)
days_since_lp = rng.integers(1, 365, N)
satisfaction  = rng.integers(1, 6, N)
support_calls = rng.integers(0, 11, N)
returns_count = rng.integers(0, (num_orders.clip(max=10)//2 + 1))
payment       = wchoice(["UPI","Credit Card","Debit Card","Net Banking","COD","Wallet"], [30,25,20,10,10,5], N)
membership    = wchoice(["Silver","Gold","Platinum"], [55,30,15], N)

CATS = ["Electronics","Fashion","Home_Kitchen","Beauty","Books","Sports","Toys"]
pref_cat = wchoice(CATS, [25,22,18,12,10,8,5], N)
cat_cols = {}
for cat in CATS:
    base = (pref_cat == cat).astype(float) * rng.integers(5, 15, N)
    cat_cols[f"orders_{cat}"] = (base + rng.integers(0, 5, N)).astype(int)

churn_score = (
    0.30*(days_since_lp/365) + 0.20*(support_calls/10) +
    0.15*((5-satisfaction)/4) + 0.15*((72-tenure_months)/72) +
    0.10*(returns_count/10)   + 0.10*(1-num_orders/num_orders.max()) +
    rng.uniform(-0.15, 0.15, N)
).clip(0,1)
churn = (churn_score > 0.52).astype(int)

df = pd.DataFrame({
    "CustomerID": customer_ids, "Age": ages, "Gender": genders,
    "Location": locations, "Annual_Income": annual_income,
    "Tenure_Months": tenure_months, "Num_Orders": num_orders,
    "Avg_Order_Value": avg_order_val, "Total_Spend": total_spend,
    "Days_Since_Last_Purchase": days_since_lp,
    "Satisfaction_Score": satisfaction, "Support_Calls": support_calls,
    "Returns_Count": returns_count, "Preferred_Category": pref_cat,
    "Payment_Method": payment, "Membership_Tier": membership,
    **cat_cols, "Churn": churn,
})
for col in ["Annual_Income","Satisfaction_Score","Support_Calls","Avg_Order_Value"]:
    df.loc[rng.random(N) < 0.03, col] = np.nan

os.makedirs("data", exist_ok=True)
df.to_csv("data/ecommerce_customers.csv", index=False)
print(f"Dataset written  ({len(df):,} rows x {df.shape[1]} cols)  Churn: {df['Churn'].mean():.1%}")
