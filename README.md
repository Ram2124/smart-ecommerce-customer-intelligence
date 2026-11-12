# 🛒 Smart E-Commerce Customer Intelligence

> An end-to-end Data Science & Machine Learning project for customer analytics, segmentation, churn prediction, and product recommendations — built with **Streamlit**, **scikit-learn**, and **Plotly**.

---

## ✨ Features

| Module | Description |
|--------|------------|
| 📊 **EDA Overview** | Histograms, bar charts, scatter explorer, correlation heatmap |
| 👥 **Customer Segmentation** | K-Means clustering with elbow method, segment profiles, downloadable results |
| ⚠️ **Churn Prediction** | 3-model comparison (Logistic Regression, Random Forest, Gradient Boosting) with ROC-AUC, F1, and confusion matrix |
| 🛍️ **Product Recommendations** | Item-based collaborative filtering — customer-level and segment-level |
| 🔎 **Data Explorer** | Filterable data table with CSV download |

---

## 🏗️ Project Structure

```
smart-ecommerce-customer-intelligence/
├── app.py                  # Main Streamlit dashboard
├── generate_dataset.py     # Synthetic dataset generator (run once)
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml         # Dark theme configuration
├── data/
│   └── ecommerce_customers.csv
└── src/
    ├── __init__.py
    ├── data_loader.py      # CSV loading & cleaning utilities
    ├── models.py           # ML training (KMeans, churn classifiers)
    ├── recommender.py      # Cosine-similarity recommendation engine
    └── ui_components.py    # Reusable Plotly / Streamlit chart components
```

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/smart-ecommerce-customer-intelligence
cd smart-ecommerce-customer-intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the dataset (5 000 synthetic customer records)
python generate_dataset.py

# 4. Launch the app
streamlit run app.py
```

---

## 🧠 ML Stack

| Component | Algorithm | Library |
|-----------|-----------|---------|
| Segmentation | K-Means (k=2–8) | scikit-learn |
| Churn Prediction | Logistic Regression | scikit-learn |
| Churn Prediction | Random Forest (200 trees) | scikit-learn |
| Churn Prediction | Gradient Boosting (150 estimators) | scikit-learn |
| Recommendations | Cosine Similarity (item-based CF) | scikit-learn / scipy |

---

## 📊 Dataset

5 000 synthetic e-commerce customer records with:
- Demographics: Age, Gender, Location
- Engagement: Tenure, Orders, Total Spend, Days Since Last Purchase
- Satisfaction: Satisfaction Score, Support Calls, Returns
- Category purchase counts (Electronics, Fashion, Home & Kitchen, Beauty, Books, Sports, Toys)
- Membership Tier & Payment Method
- **Churn label** (rule-based with realistic ~27% churn rate)

---

## ☁️ Deployment

The app is designed for **Streamlit Community Cloud**:
1. Push repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select `app.py` as the main file
4. Deploy 🎉

---

## 👤 Author

**[Your Name]** — Data Science Portfolio Project  
[LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername)
