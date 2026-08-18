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


ANIMATED_CSS = """
<style>
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(24px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}
@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.03); }
}
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-30px); }
    to { opacity: 1; transform: translateX(0); }
}
@keyframes slideInRight {
    from { opacity: 0; transform: translateX(30px); }
    to { opacity: 1; transform: translateX(0); }
}
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-6px); }
}
.hero-banner {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 30%, #A78BFA 60%, #6366F1 100%);
    background-size: 200% 200%;
    animation: gradientShift 6s ease infinite;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 1.5rem;
    color: white;
}
.hero-banner h1 {
    color: white !important;
    font-size: 2.2rem !important;
    margin-bottom: 0.3rem !important;
    animation: fadeInUp 0.8s ease-out;
}
.hero-banner p {
    color: rgba(255,255,255,0.9) !important;
    font-size: 1.1rem !important;
    animation: fadeInUp 1s ease-out 0.2s both;
}
.fade-in {
    animation: fadeInUp 0.6s ease-out both;
}
.fade-in-delay-1 { animation-delay: 0.1s; }
.fade-in-delay-2 { animation-delay: 0.2s; }
.fade-in-delay-3 { animation-delay: 0.3s; }
.fade-in-delay-4 { animation-delay: 0.4s; }
.pulse-card:hover {
    animation: pulse 0.6s ease;
}
.float-icon {
    animation: float 3s ease-in-out infinite;
    display: inline-block;
    font-size: 2.5rem;
}
.pipeline-step {
    background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
    border-left: 4px solid #6366F1;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin: 0.8rem 0;
    animation: slideInLeft 0.5s ease-out both;
}
.pipeline-step:nth-child(1) { animation-delay: 0.1s; }
.pipeline-step:nth-child(2) { animation-delay: 0.25s; }
.pipeline-step:nth-child(3) { animation-delay: 0.4s; }
.pipeline-step:nth-child(4) { animation-delay: 0.55s; }
.pipeline-step:nth-child(5) { animation-delay: 0.7s; }
.risk-high {
    background: linear-gradient(135deg, #FEF2F2, #FEE2E2);
    border-left: 4px solid #EF4444;
    border-radius: 10px;
    padding: 1.5rem;
    animation: pulse 2s ease-in-out infinite;
}
.risk-medium {
    background: linear-gradient(135deg, #FFFBEB, #FEF3C7);
    border-left: 4px solid #F59E0B;
    border-radius: 10px;
    padding: 1.5rem;
    animation: pulse 2.5s ease-in-out infinite;
}
.risk-low {
    background: linear-gradient(135deg, #F0FDF4, #DCFCE7);
    border-left: 4px solid #10B981;
    border-radius: 10px;
    padding: 1.5rem;
    animation: pulse 3s ease-in-out infinite;
}
.stat-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid #E2E8F0;
    transition: all 0.3s ease;
    animation: fadeInUp 0.6s ease-out both;
}
.stat-card:hover {
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.15);
    transform: translateY(-2px);
}
.stat-number {
    font-size: 2rem;
    font-weight: 700;
    color: #6366F1;
    line-height: 1.2;
}
.stat-label {
    font-size: 0.85rem;
    color: #64748B;
    margin-top: 0.3rem;
}
.flow-arrow {
    text-align: center;
    font-size: 1.5rem;
    color: #6366F1;
    animation: float 2s ease-in-out infinite;
    padding: 0.3rem 0;
}
.section-badge {
    display: inline-block;
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    color: white;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
    animation: fadeIn 0.8s ease-out;
}
</style>
"""


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


st.html(ANIMATED_CSS)


st.html("""
<div class="hero-banner">
    <h1>Telecom customer churn predictor</h1>
    <p>Estimate whether a customer will cancel their subscription next month using machine learning.</p>
</div>
""")


with st.sidebar:
    st.html("""
    <div style="background:linear-gradient(135deg,#6366F1,#8B5CF6);border-radius:12px;padding:1.2rem 1rem;margin-bottom:1rem;color:white;">
        <div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;opacity:0.8;margin-bottom:0.3rem;">Model status</div>
        <div style="font-size:1rem;font-weight:600;">""" + model_source + """</div>
    </div>
    """)

    st.markdown("#### :material/info: About")
    st.write(
        "A decision-support signal for retention teams — not an automatic decision."
    )

    st.markdown("---")

    st.markdown("#### :material/gauge: Risk guidance")
    st.html("""
    <div style="display:flex;flex-direction:column;gap:0.5rem;">
        <div style="display:flex;align-items:center;gap:0.6rem;background:#FEF2F2;border-radius:8px;padding:0.6rem 0.8rem;">
            <span style="font-size:1.1rem;">&#9888;</span>
            <div><strong style="color:#DC2626;">High</strong><br><span style="font-size:0.8rem;color:#6B7280;">60% or more</span></div>
        </div>
        <div style="display:flex;align-items:center;gap:0.6rem;background:#FFFBEB;border-radius:8px;padding:0.6rem 0.8rem;">
            <span style="font-size:1.1rem;">&#9888;</span>
            <div><strong style="color:#D97706;">Medium</strong><br><span style="font-size:0.8rem;color:#6B7280;">35% - 59%</span></div>
        </div>
        <div style="display:flex;align-items:center;gap:0.6rem;background:#F0FDF4;border-radius:8px;padding:0.6rem 0.8rem;">
            <span style="font-size:1.1rem;">&#10003;</span>
            <div><strong style="color:#16A34A;">Low</strong><br><span style="font-size:0.8rem;color:#6B7280;">Below 35%</span></div>
        </div>
    </div>
    """)

    st.markdown("---")

    st.markdown("#### :material/lightbulb: Tips")
    st.info("Use realistic values for best predictions. Extreme charges with short tenure often signal high risk.")

    st.markdown("---")

    st.markdown("#### :material/link: Quick links")
    st.html("""
    <div style="display:flex;flex-direction:column;gap:0.4rem;">
        <a href="https://github.com/MubiKhanKhattak/telco-churn-predictor" target="_blank" style="display:flex;align-items:center;gap:0.5rem;padding:0.5rem 0.8rem;background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;text-decoration:none;color:#1E293B;font-size:0.9rem;transition:all 0.2s;">
            &#128279; GitHub repo
        </a>
        <a href="https://telco-churn-predictor.streamlit.app" target="_blank" style="display:flex;align-items:center;gap:0.5rem;padding:0.5rem 0.8rem;background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;text-decoration:none;color:#1E293B;font-size:0.9rem;transition:all 0.2s;">
            &#127760; Live app
        </a>
    </div>
    """)


tab_predict, tab_how, tab_docs = st.tabs(
    [":material/edit: Predict", ":material/science: How it works", ":material/article: Documentation"]
)


with tab_predict:
    st.html('<div class="section-badge">:material/edit: ENTER CUSTOMER DETAILS</div>')

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

    submitted = st.button(":material/send: Predict churn risk", type="primary", use_container_width=True)

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
            level, icon, message, css_class = "High", ":material/warning:", "Prioritize this customer for retention outreach and a tailored offer.", "risk-high"
        elif probability >= 0.35:
            level, icon, message, css_class = "Medium", ":material/info:", "Consider proactive outreach, especially if the customer has recently contacted support.", "risk-medium"
        else:
            level, icon, message, css_class = "Low", ":material/check_circle:", "No immediate retention action is indicated by this model.", "risk-low"

        st.space("medium")

        st.html(f"""
        <div class="{css_class}">
            <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.8rem;">
                <span class="float-icon">{icon.replace(':material/', '').replace(':', '')}</span>
                <h3 style="margin:0;">{level} churn risk</h3>
            </div>
            <div style="display:flex;gap:2rem;align-items:center;flex-wrap:wrap;">
                <div>
                    <div class="stat-number">{percentage:.1f}%</div>
                    <div class="stat-label">Predicted churn probability</div>
                </div>
                <div style="flex:1;min-width:200px;">
                    <p style="margin:0;color:#475569;">{message}</p>
                </div>
            </div>
        </div>
        """)

        with st.expander(":material/table_chart: View submitted customer data"):
            st.dataframe(customer, use_container_width=True, hide_index=True)


with tab_how:
    st.html('<div class="section-badge">:material/science: MODEL EXPLAINED</div>')
    st.subheader("How the model works")

    st.html("""
    <div class="pipeline-step fade-in">
        <h4 style="margin:0 0 0.5rem 0;color:#6366F1;">:material/input: Step 1: Raw customer data</h4>
        <p style="margin:0;color:#475569;">18 customer attributes are collected: profile info, services, and billing details.</p>
    </div>
    <div class="flow-arrow">:material/arrow_downward:</div>
    <div class="pipeline-step fade-in fade-in-delay-1">
        <h4 style="margin:0 0 0.5rem 0;color:#6366F1;">:material/build: Step 2: Preprocessing</h4>
        <p style="margin:0;color:#475569;">Missing values are imputed (median for numeric, most frequent for categorical). Numeric features are standardized. Categorical features are one-hot encoded.</p>
    </div>
    <div class="flow-arrow">:material/arrow_downward:</div>
    <div class="pipeline-step fade-in fade-in-delay-2">
        <h4 style="margin:0 0 0.5rem 0;color:#6366F1;">:material/smart_toy: Step 3: Model training</h4>
        <p style="margin:0;color:#475569;">Logistic Regression with balanced class weights learns the relationship between features and churn. Random Forest is also compared as an alternative.</p>
    </div>
    <div class="flow-arrow">:material/arrow_downward:</div>
    <div class="pipeline-step fade-in fade-in-delay-3">
        <h4 style="margin:0 0 0.5rem 0;color:#6366F1;">:material/bar_chart: Step 4: Evaluation</h4>
        <p style="margin:0;color:#475569;">The best model is selected by ROC-AUC. Classification report, confusion matrix, and ROC curve validate performance on unseen data.</p>
    </div>
    <div class="flow-arrow">:material/arrow_downward:</div>
    <div class="pipeline-step fade-in fade-in-delay-4">
        <h4 style="margin:0 0 0.5rem 0;color:#6366F1;">:material/send: Step 5: Prediction</h4>
        <p style="margin:0;color:#475569;">The trained pipeline scores new customers and outputs a churn probability with a risk level for the retention team.</p>
    </div>
    """)

    st.space("large")

    st.html('<div class="section-badge">:material/analytics: KEY NUMBERS</div>')
    st.subheader("Project at a glance")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.html("""<div class="stat-card fade-in"><div class="stat-number">7,044</div><div class="stat-label">Customer records</div></div>""")
    with c2:
        st.html("""<div class="stat-card fade-in fade-in-delay-1"><div class="stat-number">18</div><div class="stat-label">Input features</div></div>""")
    with c3:
        st.html("""<div class="stat-card fade-in fade-in-delay-2"><div class="stat-number">2</div><div class="stat-label">Models compared</div></div>""")
    with c4:
        st.html("""<div class="stat-card fade-in fade-in-delay-3"><div class="stat-number">~0.84</div><div class="stat-label">ROC-AUC score</div></div>""")

    st.space("large")

    c1, c2 = st.columns(2)

    with c1:
        with st.container(border=True):
            st.markdown("### :material/tune: Preprocessing details")
            st.markdown("""
            | Step | Method | Applies to |
            |------|--------|------------|
            | Imputation | Median | Numeric features |
            | Imputation | Most frequent | Categorical features |
            | Scaling | StandardScaler | Numeric features |
            | Encoding | OneHotEncoder | Categorical features |
            """)

    with c2:
        with st.container(border=True):
            st.markdown("### :material/compare: Model comparison")
            st.markdown("""
            | Model | Strengths | Weaknesses |
            |-------|-----------|------------|
            | **Logistic Regression** | Interpretable, fast, strong baseline | Linear decision boundary |
            | **Random Forest** | Captures non-linear patterns | Less interpretable, slower |
            """)

    with st.container(border=True):
        st.markdown("### :material/table_chart: Features used by the model")
        st.markdown("""
        | Category | Features |
        |----------|----------|
        | :material/person: **Profile** | Gender, senior citizen, partner, dependents, tenure |
        | :material/phone: **Phone** | Phone service, multiple lines |
        | :material/wifi: **Internet** | Internet service, online security, backup, device protection, tech support, streaming TV, streaming movies |
        | :material/receipt_long: **Billing** | Contract type, paperless billing, payment method, monthly charges, total charges |
        """)


with tab_docs:
    st.html('<div class="section-badge">:material/article: FULL DOCS</div>')
    st.subheader("Documentation")

    with st.container(border=True):
        st.markdown("### :material/info: Project overview")
        st.markdown("""
        This is an end-to-end machine learning project that predicts customer churn for a telecom company. It includes data exploration, model training, evaluation, and a deployed Streamlit web application.

        **Tech stack:** Python, scikit-learn, Streamlit, pandas, joblib

        **Dataset:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) - 7,044 customer records with 21 features.
        """)

    with st.container(border=True):
        st.markdown("### :material/terminal: How to run locally")
        st.code("""# Clone the repository
git clone https://github.com/MubiKhanKhattak/telco-churn-predictor.git
cd telco-churn-predictor

# Install dependencies
pip install -r requirements.txt

# Start the app
streamlit run app.py""", language="bash")

    with st.container(border=True):
        st.markdown("### :material/gauge: Understanding the prediction")
        st.markdown("""
        The model outputs a **churn probability** between 0% and 100%. This is interpreted as:

        - **High risk (60%+):** The customer has strong signals of leaving. The retention team should reach out with a tailored offer (discount, plan upgrade, service improvement).
        - **Medium risk (35-59%):** The customer shows some churn indicators. Proactive outreach is recommended, especially if they have recently contacted support.
        - **Low risk (below 35%):** The customer is likely to stay. No immediate action is needed, but monitoring continues.

        The threshold between high/medium/low can be adjusted based on the team's budget and the cost of retention offers.
        """)

    with st.container(border=True):
        st.markdown("### :material/lightbulb: Key churn drivers")
        st.markdown("""
        Based on the logistic regression coefficients, the most influential factors are:

        - **:material/gavel: Contract type:** Month-to-month contracts have the highest churn risk
        - **:material/schedule: Tenure:** Newer customers are more likely to leave
        - **:material/wifi: Internet service:** Fiber optic customers churn more (possibly due to price or service issues)
        - **:material/credit_card: Payment method:** Electronic check users have higher churn
        - **:material/shield: Tech support and online security:** Customers without these services are at higher risk

        These insights help the retention team prioritize which customers to contact and what offers to prepare.
        """)

    with st.container(border=True):
        st.markdown("### :material/emoji_events: Model performance")
        st.markdown("""
        The model is evaluated using:
        - **ROC-AUC:** Measures ranking quality across all thresholds
        - **Precision:** Of customers predicted to churn, how many actually did
        - **Recall:** Of customers who actually churned, how many were identified
        - **F1-score:** Harmonic mean of precision and recall

        For retention use cases, **churn recall** is particularly valuable because it measures how many at-risk customers the team can identify.
        """)

    with st.container(border=True):
        st.markdown("### :material/list: Retention list")
        st.markdown("""
        After evaluation, the model produces a **prioritized retention list** for the test set. Customers are ranked by churn probability, and those above 0.60 are flagged as high priority. This list helps the retention team allocate their outreach budget efficiently.

        The list is saved as `retention_priority_list.csv` during training.
        """)

    with st.container(border=True):
        st.markdown("### :material/folder: Project structure")
        st.code("""telco-churn-predictor/
  app.py                              Streamlit web app
  main.py                             ML training script
  telco_churn_kaggle.ipynb            Kaggle notebook
  telco_churn_prediction.ipynb        Local notebook
  archive/                            Dataset CSV
  requirements.txt                    Dependencies
  README.md                           Project docs""", language="text")
