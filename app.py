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


st.set_page_config(page_title="Churn Predictor", page_icon="📡", layout="wide")

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
    return pipeline, "Model trained from the included dataset"


try:
    model, model_source = load_or_train_model()
except Exception as error:
    st.error(f"The model could not start: {error}")
    st.stop()


st.title("📡 Telecom Customer Churn Predictor")
st.caption("Enter a customer's details to estimate their likelihood of cancelling next month.")

with st.sidebar:
    st.header("About this tool")
    st.write("The prediction is a decision-support signal for the retention team, not an automatic decision.")
    st.success(f"Model status: {model_source}")
    st.markdown("**Risk guidance**\n\n- High: 60% or more\n- Medium: 35%–59%\n- Low: below 35%")

with st.form("customer_form"):
    left, right = st.columns(2)

    with left:
        st.subheader("Customer profile")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Has partner", ["No", "Yes"])
        dependents = st.selectbox("Has dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
        phone_service = st.selectbox("Phone service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])

        st.subheader("Internet and support")
        internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])

    with right:
        st.subheader("Plan and billing")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly_charges = st.number_input("Monthly charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
        total_charges = st.number_input("Total charges ($)", min_value=0.0, max_value=10000.0, value=840.0, step=1.0)

        st.info("Tip: use the customer’s actual billing values. A month-to-month contract and electronic check payment can be useful signals for proactive outreach.")

    submitted = st.form_submit_button("Predict churn risk", type="primary", use_container_width=True)

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
        level, color, message = "High", "🔴", "Prioritize this customer for retention outreach and a tailored offer."
    elif probability >= 0.35:
        level, color, message = "Medium", "🟡", "Consider proactive outreach, especially if the customer has recently contacted support."
    else:
        level, color, message = "Low", "🟢", "No immediate retention action is indicated by this model."

    st.divider()
    metric, result = st.columns([1, 2])
    metric.metric("Predicted churn probability", f"{percentage:.1f}%")
    with result:
        st.subheader(f"{color} {level} churn risk")
        st.write(message)
    st.progress(int(round(percentage)))

    with st.expander("View submitted customer data"):
        st.dataframe(customer, use_container_width=True, hide_index=True)
