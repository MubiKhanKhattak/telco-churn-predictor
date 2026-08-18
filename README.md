# Telecom Customer Churn Predictor

An interactive Streamlit web app that predicts whether a telecom customer will cancel their subscription next month, helping retention teams prioritize outreach.

## Live Demo

**[Try the app live](https://YOUR_USERNAME-telco-churn-predictor.streamlit.app)** *(update this link after deploying)*

## Screenshots

> Add a screenshot of the app here after deploying.

## Features

- Enter 18 customer attributes through a clean two-column form
- Instant churn probability prediction with risk level (High / Medium / Low)
- Backed by a scikit-learn Logistic Regression pipeline with preprocessing
- Compares Logistic Regression vs Random Forest and selects the best by ROC-AUC
- Feature importance analysis identifies key churn drivers
- Generates a prioritized retention outreach list

## Project Structure

```
.
├── app.py                              # Streamlit web application
├── main.py                             # ML training script (Kaggle-ready)
├── telco_churn_kaggle.ipynb            # Kaggle notebook version
├── telco_churn_prediction.ipynb        # Local notebook version
├── archive/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── requirements.txt
└── README.md
```

## Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/telco-churn-predictor.git
   cd telco-churn-predictor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the app:
   ```bash
   streamlit run app.py
   ```

The app trains from the CSV automatically on first run. If you place a `telco_churn_model.joblib` file beside `app.py`, it will use the saved model instead.

## Tech Stack

- **Frontend:** Streamlit
- **ML:** scikit-learn (Logistic Regression, Random Forest, Pipeline, ColumnTransformer)
- **Data:** pandas, IBM Telco Customer Churn dataset
- **Model persistence:** joblib

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — IBM sample dataset with 7,044 customer records and 21 features.
