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
@keyframes drift {
    0% { transform: translate(0, 0) rotate(0deg); }
    33% { transform: translate(10px, -8px) rotate(2deg); }
    66% { transform: translate(-5px, 5px) rotate(-1deg); }
    100% { transform: translate(0, 0) rotate(0deg); }
}

/* ── Hero with telecom network SVG ── */
.hero-banner {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #4338CA 0%, #6366F1 25%, #8B5CF6 50%, #6366F1 75%, #4338CA 100%);
    background-size: 300% 300%;
    animation: gradientShift 8s ease infinite;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 1.5rem;
    color: white;
}
.hero-banner::before {
    content: '';
    position: absolute;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='400' viewBox='0 0 400 400'%3E%3Cg fill='none' stroke='rgba(255,255,255,0.08)' stroke-width='1.5'%3E%3Ccircle cx='80' cy='80' r='25'/%3E%3Ccircle cx='200' cy='60' r='18'/%3E%3Ccircle cx='320' cy='100' r='22'/%3E%3Ccircle cx='140' cy='200' r='30'/%3E%3Ccircle cx='300' cy='250' r='20'/%3E%3Ccircle cx='60' cy='320' r='15'/%3E%3Ccircle cx='220' cy='340' r='24'/%3E%3Ccircle cx='360' cy='360' r='12'/%3E%3Cline x1='80' y1='80' x2='200' y2='60'/%3E%3Cline x1='200' y1='60' x2='320' y2='100'/%3E%3Cline x1='80' y1='80' x2='140' y2='200'/%3E%3Cline x1='320' y1='100' x2='300' y2='250'/%3E%3Cline x1='140' y1='200' x2='300' y2='250'/%3E%3Cline x1='140' y1='200' x2='60' y2='320'/%3E%3Cline x1='300' y1='250' x2='220' y2='340'/%3E%3Cline x1='60' y1='320' x2='220' y2='340'/%3E%3C/g%3E%3Cg fill='rgba(255,255,255,0.12)'%3E%3Ccircle cx='80' cy='80' r='4'/%3E%3Ccircle cx='200' cy='60' r='3'/%3E%3Ccircle cx='320' cy='100' r='3.5'/%3E%3Ccircle cx='140' cy='200' r='5'/%3E%3Ccircle cx='300' cy='250' r='3.5'/%3E%3Ccircle cx='60' cy='320' r='3'/%3E%3Ccircle cx='220' cy='340' r='4'/%3E%3C/g%3E%3C/svg%3E");
    background-size: 400px 400px;
    animation: drift 20s ease-in-out infinite;
    pointer-events: none;
}
.hero-banner h1 {
    color: white !important;
    font-size: 2.2rem !important;
    margin-bottom: 0.3rem !important;
    animation: fadeInUp 0.8s ease-out;
    position: relative;
    z-index: 1;
}
.hero-banner p {
    color: rgba(255,255,255,0.9) !important;
    font-size: 1.1rem !important;
    animation: fadeInUp 1s ease-out 0.2s both;
    position: relative;
    z-index: 1;
}

/* ── Scenario problem card ── */
.scenario-problem {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #FFF1F2, #FFE4E6);
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    border-left: 5px solid #E11D48;
    animation: fadeInUp 0.6s ease-out both;
}
.scenario-problem::after {
    content: '';
    position: absolute;
    right: 15px;
    bottom: 10px;
    width: 120px;
    height: 120px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 120 120'%3E%3Cg fill='none' stroke='rgba(225,29,72,0.1)' stroke-width='2'%3E%3Cpath d='M60 10 L60 30 M60 90 L60 110 M10 60 L30 60 M90 60 L110 60'/%3E%3Ccircle cx='60' cy='60' r='35'/%3E%3Ccircle cx='60' cy='60' r='20'/%3E%3Cpath d='M45 50 L55 65 L75 45' stroke-width='3' stroke='%23E11D48' opacity='0.2'/%3E%3C/g%3E%3C/svg%3E");
    background-size: contain;
    background-repeat: no-repeat;
    opacity: 0.6;
    pointer-events: none;
}

/* ── Scenario solution card ── */
.scenario-solution {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #ECFDF5, #D1FAE5);
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    border-left: 5px solid #10B981;
    animation: fadeInUp 0.6s ease-out 0.15s both;
}
.scenario-solution::after {
    content: '';
    position: absolute;
    right: 15px;
    bottom: 10px;
    width: 120px;
    height: 120px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 120 120'%3E%3Cg fill='none' stroke='rgba(16,185,129,0.1)' stroke-width='2'%3E%3Ccircle cx='60' cy='60' r='40'/%3E%3Ccircle cx='60' cy='60' r='25'/%3E%3Cpath d='M42 60 L55 73 L78 47' stroke-width='4' stroke='%2310B981' opacity='0.25'/%3E%3C/g%3E%3C/svg%3E");
    background-size: contain;
    background-repeat: no-repeat;
    opacity: 0.6;
    pointer-events: none;
}

/* ── Pipeline steps with data-flow SVG ── */
.pipeline-step {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
    border-left: 4px solid #6366F1;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin: 0.8rem 0;
    animation: slideInLeft 0.5s ease-out both;
}
.pipeline-step::after {
    content: '';
    position: absolute;
    right: 10px;
    top: 50%;
    transform: translateY(-50%);
    width: 80px;
    height: 80px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 80 80'%3E%3Cg fill='none' stroke='rgba(99,102,241,0.08)' stroke-width='1.5'%3E%3Crect x='10' y='10' width='20' height='20' rx='3'/%3E%3Crect x='50' y='10' width='20' height='20' rx='3'/%3E%3Crect x='30' y='50' width='20' height='20' rx='3'/%3E%3Cline x1='30' y1='20' x2='50' y2='20'/%3E%3Cline x1='40' y1='30' x2='40' y2='50'/%3E%3Ccircle cx='40' cy='20' r='2' fill='rgba(99,102,241,0.15)'/%3E%3Ccircle cx='40' cy='40' r='2' fill='rgba(99,102,241,0.15)'/%3E%3C/g%3E%3C/svg%3E");
    background-size: contain;
    background-repeat: no-repeat;
    pointer-events: none;
}
.pipeline-step:nth-child(1) { animation-delay: 0.1s; }
.pipeline-step:nth-child(3) { animation-delay: 0.25s; }
.pipeline-step:nth-child(5) { animation-delay: 0.4s; }
.pipeline-step:nth-child(7) { animation-delay: 0.55s; }
.pipeline-step:nth-child(9) { animation-delay: 0.7s; }

/* ── Risk cards with themed SVGs ── */
.risk-high {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #FEF2F2, #FEE2E2);
    border-left: 4px solid #EF4444;
    border-radius: 10px;
    padding: 1.5rem;
    animation: pulse 2s ease-in-out infinite;
}
.risk-high::after {
    content: '';
    position: absolute;
    right: 10px;
    bottom: 8px;
    width: 100px;
    height: 100px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Cg fill='none' stroke='rgba(239,68,68,0.1)' stroke-width='2'%3E%3Cpath d='M50 15 L85 80 H15 Z'/%3E%3Ccircle cx='50' cy='55' r='4'/%3E%3Cline x1='50' y1='35' x2='50' y2='48'/%3E%3C/g%3E%3C/svg%3E");
    background-size: contain;
    background-repeat: no-repeat;
    pointer-events: none;
}
.risk-medium {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #FFFBEB, #FEF3C7);
    border-left: 4px solid #F59E0B;
    border-radius: 10px;
    padding: 1.5rem;
    animation: pulse 2.5s ease-in-out infinite;
}
.risk-medium::after {
    content: '';
    position: absolute;
    right: 10px;
    bottom: 8px;
    width: 100px;
    height: 100px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Cg fill='none' stroke='rgba(245,158,11,0.12)' stroke-width='2'%3E%3Ccircle cx='50' cy='50' r='35'/%3E%3Cline x1='50' y1='30' x2='50' y2='55'/%3E%3Ccircle cx='50' cy='67' r='3.5'/%3E%3C/g%3E%3C/svg%3E");
    background-size: contain;
    background-repeat: no-repeat;
    pointer-events: none;
}
.risk-low {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #F0FDF4, #DCFCE7);
    border-left: 4px solid #10B981;
    border-radius: 10px;
    padding: 1.5rem;
    animation: pulse 3s ease-in-out infinite;
}
.risk-low::after {
    content: '';
    position: absolute;
    right: 10px;
    bottom: 8px;
    width: 100px;
    height: 100px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Cg fill='none' stroke='rgba(16,185,129,0.12)' stroke-width='2'%3E%3Ccircle cx='50' cy='50' r='35'/%3E%3Cpath d='M35 50 L45 62 L68 38' stroke-width='3.5'/%3E%3C/g%3E%3C/svg%3E");
    background-size: contain;
    background-repeat: no-repeat;
    pointer-events: none;
}

/* ── Stat cards ── */
.stat-card {
    position: relative;
    overflow: hidden;
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid #E2E8F0;
    transition: all 0.3s ease;
    animation: fadeInUp 0.6s ease-out both;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: -20px;
    right: -20px;
    width: 80px;
    height: 80px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 80 80'%3E%3Cg fill='none' stroke='rgba(99,102,241,0.06)' stroke-width='1.5'%3E%3Ccircle cx='40' cy='40' r='30'/%3E%3Ccircle cx='40' cy='40' r='18'/%3E%3Ccircle cx='40' cy='40' r='6'/%3E%3C/g%3E%3C/svg%3E");
    background-size: contain;
    background-repeat: no-repeat;
    pointer-events: none;
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

/* ── Flow arrows ── */
.flow-arrow {
    text-align: center;
    font-size: 1.5rem;
    color: #6366F1;
    animation: float 2s ease-in-out infinite;
    padding: 0.3rem 0;
}

/* ── Section badges ── */
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

/* ── Fade-in utilities ── */
.fade-in { animation: fadeInUp 0.6s ease-out both; }
.fade-in-delay-1 { animation-delay: 0.1s; }
.fade-in-delay-2 { animation-delay: 0.2s; }
.fade-in-delay-3 { animation-delay: 0.3s; }
.fade-in-delay-4 { animation-delay: 0.4s; }

/* ── Float icon ── */
.float-icon {
    animation: float 3s ease-in-out infinite;
    display: inline-block;
    font-size: 2.5rem;
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

    st.html("""
    <div style="position:relative;overflow:hidden;background:linear-gradient(135deg,#F0F1FE,#EDE9FE);border:1px solid #DDD6FE;border-radius:12px;padding:1.2rem 1rem 0.8rem 1rem;margin-bottom:0.5rem;">
        <div style="position:absolute;right:-10px;top:-10px;width:90px;height:90px;background-image:url(&quot;data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 90 90'%3E%3Cg fill='none' stroke='rgba(99,102,241,0.07)' stroke-width='1.5'%3E%3Ccircle cx='45' cy='45' r='35'/%3E%3Ccircle cx='45' cy='45' r='20'/%3E%3Ccircle cx='45' cy='45' r='8'/%3E%3Cline x1='45' y1='10' x2='45' y2='25'/%3E%3Cline x1='45' y1='65' x2='45' y2='80'/%3E%3Cline x1='10' y1='45' x2='25' y2='45'/%3E%3Cline x1='65' y1='45' x2='80' y2='45'/%3E%3C/g%3E%3C/svg%3E&quot;);background-size:contain;background-repeat:no-repeat;pointer-events:none;"></div>
        <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.6rem;position:relative;z-index:1;">
            <span style="font-size:1.3rem;">&#128202;</span>
            <span style="font-size:0.95rem;font-weight:600;color:#1E293B;">About this tool</span>
        </div>
        <p style="margin:0 0 0.7rem 0;font-size:0.82rem;color:#475569;line-height:1.5;position:relative;z-index:1;">Predicts customer churn using a trained Logistic Regression model on 7,044 telecom records.</p>
        <div style="display:flex;flex-wrap:wrap;gap:0.4rem;position:relative;z-index:1;">
            <span style="background:rgba(99,102,241,0.1);color:#4F46E5;font-size:0.7rem;font-weight:600;padding:0.2rem 0.6rem;border-radius:12px;">Python</span>
            <span style="background:rgba(99,102,241,0.1);color:#4F46E5;font-size:0.7rem;font-weight:600;padding:0.2rem 0.6rem;border-radius:12px;">scikit-learn</span>
            <span style="background:rgba(99,102,241,0.1);color:#4F46E5;font-size:0.7rem;font-weight:600;padding:0.2rem 0.6rem;border-radius:12px;">Streamlit</span>
            <span style="background:rgba(99,102,241,0.1);color:#4F46E5;font-size:0.7rem;font-weight:600;padding:0.2rem 0.6rem;border-radius:12px;">pandas</span>
            <span style="background:rgba(99,102,241,0.1);color:#4F46E5;font-size:0.7rem;font-weight:600;padding:0.2rem 0.6rem;border-radius:12px;">joblib</span>
        </div>
    </div>
    """)

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

    sc1, sc2 = st.columns(2)
    with sc1:
        st.html("""
        <div class="scenario-problem">
            <div style="position:relative;z-index:1;">
                <h4 style="margin:0 0 0.4rem 0;color:#BE123C;">&#9888; The Problem</h4>
                <p style="margin:0;font-size:0.88rem;color:#9F1239;line-height:1.55;">Telecom companies lose <strong>billions annually</strong> when customers cancel. Identifying at-risk customers <em>before</em> they leave is critical — but manual outreach is slow and wasteful.</p>
            </div>
        </div>
        """)
    with sc2:
        st.html("""
        <div class="scenario-solution">
            <div style="position:relative;z-index:1;">
                <h4 style="margin:0 0 0.4rem 0;color:#047857;">&#10003; The Solution</h4>
                <p style="margin:0;font-size:0.88rem;color:#065F46;line-height:1.55;">A <strong>machine learning model</strong> scores each customer's churn probability. The retention team can then focus offers and outreach on those most likely to leave.</p>
            </div>
        </div>
        """)

    st.space("small")

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
