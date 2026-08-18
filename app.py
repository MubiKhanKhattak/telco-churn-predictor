"""Interactive Streamlit app for Telecom Customer Churn prediction.

Run locally with: streamlit run app.py
"""

from pathlib import Path
import glob

import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(
    page_title="Churn Predictor",
    page_icon=":material/analytics:",
    layout="wide",
)

DATA_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = Path("telco_churn_model.joblib")
RANDOM_STATE = 42


@st.cache_resource
def load_or_train_model():
    """Load the saved pipeline, or train one from the included CSV once."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH), "Saved model"

    possible_paths = [Path("archive") / DATA_FILE, Path(DATA_FILE)]
    possible_paths += [Path(path) for path in glob.glob(f"**/{DATA_FILE}", recursive=True)]
    data_path = next((path for path in possible_paths if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError(
            f"Could not find {DATA_FILE}. Put it in an archive folder or beside app.py."
        )

    data = pd.read_csv(data_path)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    X = data.drop(columns=["Churn", "customerID"])
    y = data["Churn"].map({"Yes": 1, "No": 0})

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()
    preprocessor = ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
        ]
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline, "Trained from dataset"


try:
    model, model_source = load_or_train_model()
except Exception as error:
    st.error(f"The model could not start: {error}")
    st.stop()


st.title(":material/analytics: Telecom customer churn predictor")
st.caption("Estimate whether a customer will cancel their subscription next month.")

with st.sidebar:
    st.header(":material/info: About")
    st.write(
        "This tool is a decision-support signal for retention teams, not an automatic decision."
    )
    st.success(f":material/check_circle: Model status: {model_source}")
    st.markdown(
        """**Risk guidance**
- :red-badge[High] 60% or more
- :orange-badge[Medium] 35% - 59%
- :green-badge[Low] Below 35%"""
    )

tab_predict, tab_how, tab_docs = st.tabs(
    [":material/edit: Predict", ":material/science: How it works", ":material/article: Documentation"]
)


with tab_predict:
    with st.form("customer_form"):
        left, right = st.columns(2)

        with left:
            with st.container(border=True):
                st.markdown("**:material/person: Customer profile**")
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior_citizen = st.selectbox("Senior citizen", ["No", "Yes"])
                partner = st.selectbox("Has partner", ["No", "Yes"])
                dependents = st.selectbox("Has dependents", ["No", "Yes"])
                tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)

            with st.container(border=True):
                st.markdown("**:material/phone: Phone and internet**")
                phone_service = st.selectbox("Phone service", ["Yes", "No"])
                multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
                internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
                online_security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
                online_backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])

        with right:
            with st.container(border=True):
                st.markdown("**:material/devices: Support and streaming**")
                device_protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
                tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])

            with st.container(border=True):
                st.markdown("**:material/receipt_long: Plan and billing**")
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
                payment_method = st.selectbox(
                    "Payment method",
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                )
                monthly_charges = st.number_input("Monthly charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
                total_charges = st.number_input("Total charges ($)", min_value=0.0, max_value=10000.0, value=840.0, step=1.0)

        submitted = st.form_submit_button(":material/send: Predict churn risk", type="primary", use_container_width=True)

    if submitted:
        customer = pd.DataFrame(
            [
                {
                    "gender": gender,
                    "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
                    "Partner": partner,
                    "Dependents": dependents,
                    "tenure": tenure,
                    "PhoneService": phone_service,
                    "MultipleLines": multiple_lines,
                    "InternetService": internet_service,
                    "OnlineSecurity": online_security,
                    "OnlineBackup": online_backup,
                    "DeviceProtection": device_protection,
                    "TechSupport": tech_support,
                    "StreamingTV": streaming_tv,
                    "StreamingMovies": streaming_movies,
                    "Contract": contract,
                    "PaperlessBilling": paperless_billing,
                    "PaymentMethod": payment_method,
                    "MonthlyCharges": monthly_charges,
                    "TotalCharges": total_charges,
                }
            ]
        )
        probability = float(model.predict_proba(customer)[0, 1])
        percentage = probability * 100

        if probability >= 0.60:
            level, icon, color, message = "High", ":material/warning:", "red", "Prioritize this customer for retention outreach and a tailored offer."
        elif probability >= 0.35:
            level, icon, color, message = "Medium", ":material/info:", "orange", "Consider proactive outreach, especially if the customer has recently contacted support."
        else:
            level, icon, color, message = "Low", ":material/check_circle:", "green", "No immediate retention action is indicated by this model."

        st.space("medium")

        with st.container(border=True):
            st.subheader(f"{icon} {level} churn risk")
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Predicted churn probability", f"{percentage:.1f}%")
                st.progress(int(round(percentage)))
            with c2:
                st.write(message)
                with st.expander(":material/table_chart: View submitted customer data"):
                    st.dataframe(customer, use_container_width=True, hide_index=True)


with tab_how:
    st.subheader(":material/science: How the model works")

    with st.container(border=True):
        st.markdown(
            """### Data pipeline

The model uses a **scikit-learn Pipeline** that chains preprocessing and classification into a single object. This prevents data leakage because all transformations are learned from training data only."""
        )

    c1, c2 = st.columns(2)

    with c1:
        with st.container(border=True):
            st.markdown(
                """### Preprocessing steps

1. **Missing values** are filled using the median (numeric) or most frequent value (categorical)
2. **Numeric features** are standardized with zero mean and unit variance
3. **Categorical features** are one-hot encoded, with unknown categories handled gracefully"""
            )

    with c2:
        with st.container(border=True):
            st.markdown(
                """### Model selection

Two models are compared:
- **Logistic Regression** - interpretable baseline with balanced class weights
- **Random Forest** - ensemble of 400 trees capturing non-linear patterns

The winner is selected by **ROC-AUC**, which measures how well the model ranks churners above non-churners across all probability thresholds."""
            )

    with st.container(border=True):
        st.markdown(
            """### Features used

The model uses **18 customer attributes** across four categories:

| Category | Features |
|----------|----------|
| **Profile** | Gender, senior citizen, partner, dependents, tenure |
| **Phone** | Phone service, multiple lines |
| **Internet** | Internet service, online security, backup, device protection, tech support, streaming TV, streaming movies |
| **Billing** | Contract type, paperless billing, payment method, monthly charges, total charges |"""
        )


with tab_docs:
    st.subheader(":material/article: Documentation")

    with st.container(border=True):
        st.markdown(
            """### Project overview

This is an end-to-end machine learning project that predicts customer churn for a telecom company. It includes data exploration, model training, evaluation, and a deployed Streamlit web application.

**Tech stack:** Python, scikit-learn, Streamlit, pandas, joblib

**Dataset:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) - 7,044 customer records with 21 features."""
        )

    with st.container(border=True):
        st.markdown(
            """### How to run locally

```bash
# Clone the repository
git clone https://github.com/MubiKhanKhattak/telco-churn-predictor.git
cd telco-churn-predictor

# Install dependencies
pip install -r requirements.txt

# Start the app
streamlit run app.py
```"""
        )

    with st.container(border=True):
        st.markdown(
            """### Understanding the prediction

The model outputs a **churn probability** between 0% and 100%. This is interpreted as:

- **High risk (60%+):** The customer has strong signals of leaving. The retention team should reach out with a tailored offer (discount, plan upgrade, service improvement).
- **Medium risk (35-59%):** The customer shows some churn indicators. Proactive outreach is recommended, especially if they have recently contacted support.
- **Low risk (below 35%):** The customer is likely to stay. No immediate action is needed, but monitoring continues.

The threshold between high/medium/low can be adjusted based on the team's budget and the cost of retention offers."""
        )

    with st.container(border=True):
        st.markdown(
            """### Key churn drivers

Based on the logistic regression coefficients, the most influential factors are:

- **Contract type:** Month-to-month contracts have the highest churn risk
- **Tenure:** Newer customers are more likely to leave
- **Internet service:** Fiber optic customers churn more (possibly due to price or service issues)
- **Payment method:** Electronic check users have higher churn
- **Tech support and online security:** Customers without these services are at higher risk

These insights help the retention team prioritize which customers to contact and what offers to prepare."""
        )

    with st.container(border=True):
        st.markdown(
            """### Model performance

The model is evaluated using:
- **ROC-AUC:** Measures ranking quality across all thresholds
- **Precision:** Of customers predicted to churn, how many actually did
- **Recall:** Of customers who actually churned, how many were identified
- **F1-score:** Harmonic mean of precision and recall

For retention use cases, **churn recall** is particularly valuable because it measures how many at-risk customers the team can identify."""
        )

    with st.container(border=True):
        st.markdown(
            """### Retention list

After evaluation, the model produces a **prioritized retention list** for the test set. Customers are ranked by churn probability, and those above 0.60 are flagged as high priority. This list helps the retention team allocate their outreach budget efficiently.

The list is saved as `retention_priority_list.csv` during training."""
        )

    with st.container(border=True):
        st.markdown(
            """### Project structure

```
telco-churn-predictor/
  app.py                              Streamlit web app
  main.py                             ML training script
  telco_churn_kaggle.ipynb            Kaggle notebook
  telco_churn_prediction.ipynb        Local notebook
  archive/                            Dataset CSV
  requirements.txt                    Dependencies
  README.md                           Project docs
```"""
        )
