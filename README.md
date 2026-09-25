# Smart E-Commerce Customer Intelligence

An end-to-end Data Science and Machine Learning project for customer analytics, segmentation, churn prediction, and business insights.

## Features
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Customer segmentation with K-Means
- Churn prediction with classification models
- Interactive Streamlit dashboard
- Customer-level prediction interface
- Model evaluation and downloadable insights

## Project Structure
```text
smart-ecommerce-customer-intelligence/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── data/
│   └── ecommerce_customers.csv
└── src/
    └── preprocessing.py
```

## Run locally
```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

## Deployment
The app is designed for Streamlit Community Cloud. Upload the repository, select `app.py` as the main file, and deploy.

> Dataset and model metrics shown by the application are calculated from the supplied dataset; no synthetic performance claims are used.
