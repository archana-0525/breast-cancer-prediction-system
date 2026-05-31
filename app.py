import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from io import BytesIO
import hashlib
import re

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Breast Cancer Prediction System",
    page_icon="🎗️",
    layout="wide"
)

# =========================
# Session State
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = "User"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# =========================
# Custom CSS
# =========================
# Stable professional theme: no manual theme switch, so the app loads faster and text remains visible.
st.markdown("""
<style>
:root {
    --app-text: #1f2937;
    --muted-text: #37474f;
    --primary: #0d47a1;
    --pink: #ec407a;
    --soft-pink: #fce4ec;
    --card-bg: rgba(255,255,255,0.96);
}

.stApp {
    background: linear-gradient(135deg, #e3f2fd 0%, #fce4ec 45%, #ffffff 100%);
}

/* Keep text readable everywhere */
h1, h2, h3, h4, h5, h6,
p, span, label,
[data-testid="stMarkdownContainer"],
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] * {
    color: var(--app-text) !important;
}

/* Sidebar visibility */
section[data-testid="stSidebar"] {
    background: #f8fafc !important;
}

section[data-testid="stSidebar"] *,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: #111827 !important;
    font-weight: 600;
}

/* Header */
.main-title {
    font-size: 44px;
    font-weight: 900;
    color: var(--primary) !important;
    text-align: center;
    margin-bottom: 8px;
}

.subtitle {
    font-size: 18px;
    text-align: center;
    color: var(--muted-text) !important;
    margin-bottom: 30px;
}

/* HTML wrappers should not create empty boxes */
.card,
.login-card {
    background: transparent !important;
    padding: 0 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    margin-bottom: 22px;
    border: none !important;
}

/* Streamlit cards/containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--card-bg) !important;
    border-radius: 18px !important;
    border: 1px solid rgba(236,64,122,0.18) !important;
    box-shadow: 0px 6px 22px rgba(0,0,0,0.10) !important;
}

/* Inputs */
input, textarea {
    color: #111827 !important;
    background-color: #ffffff !important;
}

/* Theme-safe buttons, dropdowns, and form controls */
.stButton > button,
.stDownloadButton > button,
button[kind="primary"],
button[kind="secondary"],
[data-testid="baseButton-secondary"],
[data-testid="baseButton-primary"] {
    background-color: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #ec407a !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover,
[data-testid="baseButton-secondary"]:hover,
[data-testid="baseButton-primary"]:hover {
    background-color: #fce4ec !important;
    color: #0d47a1 !important;
    border-color: #ec407a !important;
}

/* Selectbox and dropdown visibility in both browser/Streamlit themes */
[data-baseweb="select"] > div,
[data-baseweb="select"] span,
[data-baseweb="popover"],
[data-baseweb="menu"],
[data-baseweb="menu"] li,
div[role="listbox"],
div[role="option"] {
    background-color: #ffffff !important;
    color: #111827 !important;
}

/* Disabled input text visibility */
input:disabled,
textarea:disabled {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    background-color: #f8fafc !important;
    opacity: 1 !important;
}

/* Feedback section */
.feedback-box {
    background: linear-gradient(135deg, #ffffff 0%, #fff3f8 100%);
    padding: 26px;
    border-radius: 20px;
    border: 1px solid rgba(236,64,122,0.22);
    box-shadow: 0px 6px 20px rgba(0,0,0,0.08);
}

.feedback-box h3 {
    color: #ad1457 !important;
}

.feedback-box p {
    color: #263238 !important;
    line-height: 1.7;
}

/* Result boxes */
.result-benign {
    background: linear-gradient(135deg, #c8e6c9, #e8f5e9);
    padding: 28px;
    border-radius: 20px;
    text-align: center;
    color: #1b5e20 !important;
    font-size: 28px;
    font-weight: 850;
    border: 2px solid #66bb6a;
}

.result-benign span {
    color: #1b5e20 !important;
}

.result-malignant {
    background: linear-gradient(135deg, #ffcdd2, #ffebee);
    padding: 28px;
    border-radius: 20px;
    text-align: center;
    color: #b71c1c !important;
    font-size: 28px;
    font-weight: 850;
    border: 2px solid #ef5350;
}

.result-malignant span {
    color: #b71c1c !important;
}

/* Info and warning boxes */
.info-box {
    background-color: rgba(255,255,255,0.95) !important;
    padding: 18px;
    border-radius: 15px;
    border-left: 6px solid #1976d2;
    margin-top: 18px;
    color: #263238 !important;
}

.info-box * {
    color: #263238 !important;
}

.warning-box {
    background-color: #ffebee !important;
    padding: 18px;
    border-radius: 15px;
    border-left: 6px solid #c62828;
    margin-top: 18px;
    color: #263238 !important;
}

.warning-box * {
    color: #263238 !important;
}

.badge {
    display: inline-block;
    padding: 8px 14px;
    border-radius: 25px;
    background-color: var(--soft-pink);
    color: #ad1457 !important;
    font-weight: 700;
    margin-right: 8px;
    margin-bottom: 8px;
}

.chat-question {
    background-color: #e3f2fd;
    padding: 14px;
    border-radius: 15px;
    color: #0d47a1 !important;
    font-weight: 700;
    margin-top: 12px;
}

.chat-answer {
    background-color: #fff3f8;
    padding: 16px;
    border-radius: 15px;
    color: #37474f !important;
    border-left: 6px solid #ec407a;
    margin-top: 10px;
    line-height: 1.6;
}

.chat-answer * {
    color: #37474f !important;
}

/* Professional About section */
.about-hero {
    background: linear-gradient(135deg, #ffffff 0%, #fff3f8 100%);
    padding: 30px;
    border-radius: 22px;
    border: 1px solid rgba(236,64,122,0.18);
    box-shadow: 0px 6px 22px rgba(0,0,0,0.08);
    margin-bottom: 22px;
}

.about-hero h2 {
    color: #0d47a1 !important;
    margin-bottom: 14px;
}

.about-hero p {
    font-size: 17px;
    line-height: 1.8;
    color: #263238 !important;
}

.about-section {
    background-color: rgba(255,255,255,0.96);
    padding: 22px;
    border-radius: 18px;
    border-left: 6px solid #ec407a;
    margin-top: 18px;
    box-shadow: 0px 4px 16px rgba(0,0,0,0.06);
}

.about-section h3 {
    color: #ad1457 !important;
    margin-bottom: 10px;
}

.about-section p {
    font-size: 16px;
    line-height: 1.7;
    color: #263238 !important;
}

.password-rule {
    background-color: #f8fafc;
    padding: 12px 14px;
    border-radius: 12px;
    border-left: 5px solid #ec407a;
    color: #263238 !important;
    margin-top: 12px;
    font-size: 14px;
    line-height: 1.6;
}

</style>
""", unsafe_allow_html=True)

# =========================
# File Paths
# =========================
MODEL_PATH = Path("xgboost_breast_cancer_model.pkl")
DATA_PATH = Path("breast_cancer_dataframe.csv")
CLEAN_DATA_PATH = Path("cleaned_breast_cancer_dataset.csv")
HISTORY_PATH = Path("prediction_history.csv")
USER_PATH = Path("users.csv")
FEEDBACK_PATH = Path("feedback.csv")


# =========================
# Authentication Helpers
# =========================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def is_valid_password(password):
    """
    Password rules:
    - Minimum 6 characters
    - At least 1 uppercase letter
    - At least 1 special character
    - Lowercase letters and digits are allowed
    """
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter."
    if not re.search(r"[^A-Za-z0-9]", password):
        return False, "Password must contain at least one special character such as @, #, $, %, or !."
    return True, "Valid password."

def load_users():
    if USER_PATH.exists():
        return pd.read_csv(USER_PATH)
    return pd.DataFrame(columns=["username", "password_hash", "created_at"])

def save_user(username, password):
    users_df = load_users()
    new_user = pd.DataFrame([{
        "username": username.strip().lower(),
        "password_hash": hash_password(password),
        "created_at": datetime.now().strftime("%d-%m-%Y %I:%M:%S %p")
    }])
    users_df = pd.concat([users_df, new_user], ignore_index=True)
    users_df.to_csv(USER_PATH, index=False)

def authenticate_user(username, password):
    users_df = load_users()
    username_key = username.strip().lower()

    if users_df.empty or username_key not in users_df["username"].astype(str).values:
        return False, "Username not found. Please create an account first."

    stored_hash = users_df.loc[users_df["username"] == username_key, "password_hash"].iloc[0]
    if hash_password(password) == stored_hash:
        return True, "Login successful."

    return False, "Invalid password. Please enter the correct password."


# =========================
# Login System
# =========================
def login_page():
    st.markdown("<div class='main-title'>🎗️ Breast Cancer Prediction System</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Secure AI-powered tumor classification platform</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.25, 1])
    with col2:
        st.markdown("<div class='login-card'>", unsafe_allow_html=True)
        st.subheader("Secure Account Access")
        st.write("Login with your registered credentials or create a new account to access the prediction dashboard.")

        auth_mode = st.radio("Choose Option", ["Login", "Create Account"], horizontal=True)

        username = st.text_input("Username", placeholder="Example: Secure")
        password = st.text_input("Password", type="password", placeholder="Enter password")

        st.markdown("""
        <div class='password-rule'>
        <b>Password Requirements</b><br>
        • Minimum 6 characters<br>
        • At least 1 uppercase letter (A-Z)<br>
        • At least 1 special character (@, #, $, %, !, etc.)<br>
        • Lowercase letters and digits are allowed<br><br>
        <b>Example:</b> Secure@123
        </div>
        """, unsafe_allow_html=True)

        if auth_mode == "Login":
            if st.button("Login to Dashboard", use_container_width=True):
                if username.strip() == "" or password.strip() == "":
                    st.error("Please enter both username and password.")
                else:
                    success, message = authenticate_user(username, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.username = username.strip()
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)

        else:
            if st.button("Create Account", use_container_width=True):
                if username.strip() == "" or password.strip() == "":
                    st.error("Please enter both username and password.")
                else:
                    valid, message = is_valid_password(password)
                    users_df = load_users()
                    username_key = username.strip().lower()

                    if not valid:
                        st.error(message)
                    elif username_key in users_df["username"].astype(str).values:
                        st.error("Username already exists. Please login instead.")
                    else:
                        save_user(username, password)
                        st.success("Account created successfully. Now login using the same credentials.")

        st.markdown("</div>", unsafe_allow_html=True)

if not st.session_state.logged_in:
    login_page()
    st.stop()

# =========================
# Load Model
# =========================
if not MODEL_PATH.exists():
    st.error("Model file not found. Keep xgboost_breast_cancer_model.pkl in the same folder as app.py.")
    st.stop()
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# =========================
# Feature Names
# =========================
feature_names = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]
important_features = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "worst radius", "worst concave points"
]

# =========================
# Load Dataset
# =========================
try:
    if CLEAN_DATA_PATH.exists():
        df = pd.read_csv(CLEAN_DATA_PATH)
    else:
        df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error("Dataset file not found. Keep breast_cancer_dataframe.csv or cleaned_breast_cancer_dataset.csv in the same folder as app.py.")
    st.stop()

if "Unnamed: 0" in df.columns:
    df = df.drop("Unnamed: 0", axis=1)
if "target" not in df.columns:
    st.error("Target column is missing from the dataset.")
    st.stop()

X = df.drop("target", axis=1)
default_values = X.median().to_dict()
mean_values = X.mean().to_dict()

expected_features = getattr(model, "n_features_in_", 30)
if expected_features != 30:
    st.error(f"Your saved model expects {expected_features} features, but this app sends 30 features. Retrain and save the XGBoost model using 30 columns only.")
    st.stop()

# =========================
# Helper Functions
# =========================
def risk_label_and_color(malignant_prob):
    if malignant_prob >= 70:
        return "High Risk", 1.0
    elif malignant_prob >= 40:
        return "Moderate Risk", 0.6
    return "Low Risk", 0.25

def save_prediction_history(row):
    new_df = pd.DataFrame([row])
    if HISTORY_PATH.exists():
        old_df = pd.read_csv(HISTORY_PATH)
        final_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        final_df = new_df
    final_df.to_csv(HISTORY_PATH, index=False)

def save_feedback(row):
    new_df = pd.DataFrame([row])
    if FEEDBACK_PATH.exists():
        old_df = pd.read_csv(FEEDBACK_PATH)
        final_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        final_df = new_df
    final_df.to_csv(FEEDBACK_PATH, index=False)

def generate_pdf_report(case_summary, prediction_text, confidence, recommendations):
    if not REPORTLAB_AVAILABLE:
        return None
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Breast Cancer Prediction Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Prediction: {prediction_text}", styles["Heading2"]))
    story.append(Paragraph(f"Confidence: {confidence}", styles["Normal"]))
    story.append(Spacer(1, 12))

    data = [["Field", "Value"]] + [[str(k), str(v)] for k, v in case_summary.items()]
    table = Table(data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f8bbd0")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    story.append(table)
    story.append(Spacer(1, 14))
    story.append(Paragraph("Recommended Next Steps", styles["Heading2"]))
    for rec in recommendations:
        story.append(Paragraph(f"• {rec}", styles["Normal"]))
    doc.build(story)
    buffer.seek(0)
    return buffer

def show_radar_chart(input_data):
    if plt is None:
        st.info("Install matplotlib to view radar chart.")
        return
    selected = important_features
    values = np.array([float(input_data[f]) for f in selected])
    medians = np.array([float(default_values[f]) for f in selected])
    normalized = values / np.maximum(medians, 1e-8)
    normalized = np.clip(normalized, 0, 2)
    angles = np.linspace(0, 2 * np.pi, len(selected), endpoint=False).tolist()
    normalized = np.concatenate((normalized, [normalized[0]]))
    angles += angles[:1]
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, normalized, linewidth=2)
    ax.fill(angles, normalized, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f.replace("mean ", "M ").replace("worst ", "W ") for f in selected], fontsize=8)
    ax.set_yticklabels([])
    ax.set_title("Important Feature Pattern")
    st.pyplot(fig)

def chatbot_response(question):
    q = question.lower()
    if "benign" in q or "low risk" in q:
        return "Benign means the tumor pattern is classified as non-cancerous. A low-risk result usually indicates that the entered tumor characteristics are closer to benign patterns."
    elif "malignant" in q or "high risk" in q:
        return "Malignant means the tumor pattern is classified as cancerous. A high-risk result means the entered tumor measurements are closer to malignant patterns. Further clinical evaluation is recommended."
    elif "recommend" in q or "what should i do" in q or "advice" in q:
        return "For high risk, consult an oncologist or qualified specialist, carry reports, avoid delaying evaluation, and follow advice for confirmatory tests. For low risk, continue routine checkups."
    elif "feature importance" in q:
        return "Feature importance shows which tumor measurements influenced the model more strongly during prediction. Higher importance features usually contribute more to classification."
    elif "history" in q:
        return "Prediction History stores previous prediction records with timestamp, patient name or ID, age, result, and confidence."
    elif "mean radius" in q or "radius" in q:
        return "Mean radius is the average distance from the center of the tumor cell nucleus to its boundary points."
    elif "texture" in q:
        return "Mean texture measures variation in grayscale intensity of the tumor image."
    elif "concavity" in q or "concave" in q:
        return "Concavity measures inward curves on the tumor boundary. Concave points are important for classification."
    elif "confidence" in q or "probability" in q:
        return "Confidence score shows how strongly the model supports its predicted class based on the entered measurements."
    elif "age" in q or "patient" in q:
        return "Patient name and age are record fields only. They do not affect model prediction."
    elif "features" in q or "input" in q:
        return "Simple Mode uses mean radius, mean texture, mean perimeter, mean area, mean smoothness, mean compactness, mean concavity, mean concave points, worst radius, and worst concave points."
    return "I can help with benign/malignant meaning, risk levels, feature meanings, confidence score, recommendations, history, and reports."

# =========================
# Sidebar
# =========================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Prediction", "History", "Feature Guide", "Chatbot", "Feedback", "About"]
)
st.sidebar.markdown("---")
st.sidebar.write(f"Logged in as: **{st.session_state.username}**")
st.sidebar.write("Status: **Active Session**")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# =========================
# Header
# =========================
st.markdown("<div class='main-title'>🎗️ Breast Cancer Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Analyze tumor characteristics and classify them as Benign or Malignant</div>", unsafe_allow_html=True)

# =========================
# Prediction Page
# =========================
if page == "Prediction":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Patient Information")
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        patient_name = st.text_input(
            "Patient Name / ID",
            placeholder="Example: Ramya or PT-1001"
        )
    with pcol2:
        patient_age = st.number_input("Age", min_value=1, max_value=120, value=40)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Tumor Feature Input")
    input_mode = st.radio("Choose Input Mode", ["Simple Mode - Important Features", "Advanced Mode - All Features"], horizontal=True)
    input_data = default_values.copy()

    if input_mode == "Simple Mode - Important Features":
        st.write("Enter the main tumor measurement values. Other features are filled automatically using training data median values.")
        col1, col2 = st.columns(2)
        with col1:
            input_data["mean radius"] = st.number_input("Mean Radius", value=float(default_values["mean radius"]), format="%.4f")
            input_data["mean texture"] = st.number_input("Mean Texture", value=float(default_values["mean texture"]), format="%.4f")
            input_data["mean perimeter"] = st.number_input("Mean Perimeter", value=float(default_values["mean perimeter"]), format="%.4f")
            input_data["mean area"] = st.number_input("Mean Area", value=float(default_values["mean area"]), format="%.4f")
            input_data["mean smoothness"] = st.number_input("Mean Smoothness", value=float(default_values["mean smoothness"]), format="%.4f")
        with col2:
            input_data["mean compactness"] = st.number_input("Mean Compactness", value=float(default_values["mean compactness"]), format="%.4f")
            input_data["mean concavity"] = st.number_input("Mean Concavity", value=float(default_values["mean concavity"]), format="%.4f")
            input_data["mean concave points"] = st.number_input("Mean Concave Points", value=float(default_values["mean concave points"]), format="%.4f")
            input_data["worst radius"] = st.number_input("Worst Radius", value=float(default_values["worst radius"]), format="%.4f")
            input_data["worst concave points"] = st.number_input("Worst Concave Points", value=float(default_values["worst concave points"]), format="%.4f")
        st.markdown("<div class='info-box'>Simple Mode keeps the interface clean by asking only the most important tumor-related measurements.</div>", unsafe_allow_html=True)
    else:
        st.write("Enter all 30 tumor measurement features.")
        cols = st.columns(3)
        for index, feature in enumerate(feature_names):
            with cols[index % 3]:
                input_data[feature] = st.number_input(feature.title(), value=float(default_values[feature]), format="%.4f")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Predict Tumor Type", use_container_width=True):
        final_input = np.array([input_data[feature] for feature in feature_names]).reshape(1, -1)
        prediction = model.predict(final_input)[0]
        probability = model.predict_proba(final_input)[0]
        malignant_prob = probability[0] * 100
        benign_prob = probability[1] * 100
        prediction_text = "Benign - Low Risk" if prediction == 1 else "Malignant - High Risk"
        confidence = max(benign_prob, malignant_prob)
        timestamp = datetime.now().strftime("%d-%m-%Y %I:%M:%S %p")

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Prediction Result")
        if prediction == 1:
            st.markdown(f"""
            <div class='result-benign'>
                ✓ Low Risk<br><br>
                Tumor Classification: Benign<br><br>
                <span style='font-size:18px; font-weight:500;'>The analyzed tumor characteristics indicate a benign (non-cancerous) condition.</span><br><br>
                Confidence Score: {benign_prob:.2f}%
            </div>
            """, unsafe_allow_html=True)
            recommendations = [
                "Continue regular health checkups.",
                "Keep previous reports for future comparison.",
                "Monitor any new symptoms or changes if advised by a healthcare professional.",
                "Maintain routine screening as recommended for your age and health profile."
            ]
        else:
            st.markdown(f"""
            <div class='result-malignant'>
                ⚠ High Risk<br><br>
                Tumor Classification: Malignant<br><br>
                <span style='font-size:18px; font-weight:500;'>The analyzed tumor characteristics indicate a malignant (cancerous) condition.</span><br><br>
                Confidence Score: {malignant_prob:.2f}%
            </div>
            """, unsafe_allow_html=True)
            recommendations = [
                "Consult an oncologist or qualified medical specialist as soon as possible.",
                "Carry the biopsy/pathology report and tumor measurement details.",
                "Do not delay further clinical evaluation.",
                "Follow the doctor’s advice for confirmatory tests and treatment planning."
            ]

        st.write("### Risk Meter")
        risk_label, risk_score = risk_label_and_color(malignant_prob)
        st.progress(risk_score)
        st.write(f"Risk Level Based on Malignant Probability: **{risk_label}**")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Benign Probability", f"{benign_prob:.2f}%")
        m2.metric("Malignant Probability", f"{malignant_prob:.2f}%")
        m3.metric("Features Analyzed", "30")
        m4.metric("Timestamp", timestamp)

        col1, col2 = st.columns(2)
        with col1:
            st.write("### Probability Analysis")
            prob_df = pd.DataFrame({"Tumor Type": ["Malignant", "Benign"], "Probability": [malignant_prob, benign_prob]})
            st.bar_chart(prob_df.set_index("Tumor Type"))
        with col2:
            st.write("### Important Feature Pattern")
            show_radar_chart(input_data)

        st.write("### Tumor Feature Comparison")
        compare_df = pd.DataFrame({
            "Feature": important_features,
            "Entered Value": [input_data[f] for f in important_features],
            "Dataset Average": [mean_values[f] for f in important_features],
            "Dataset Median": [default_values[f] for f in important_features]
        })
        st.dataframe(compare_df, use_container_width=True)

        st.write("### Case Summary")
        case_summary = {
            "Patient Name / ID": patient_name if patient_name else "Not provided",
            "Age": patient_age,
            "Prediction": prediction_text,
            "Confidence": f"{confidence:.2f}%",
            "Prediction Time": timestamp
        }
        st.dataframe(pd.DataFrame({"Field": list(case_summary.keys()), "Value": list(case_summary.values())}), use_container_width=True)

        st.write("### Recommended Next Steps")
        box_class = "info-box" if prediction == 1 else "warning-box"
        rec_html = "<br>".join([f"• {r}" for r in recommendations])
        st.markdown(f"<div class='{box_class}'><b>{'Low Risk' if prediction == 1 else 'High Risk'} Recommendation</b><br>{rec_html}</div>", unsafe_allow_html=True)

        history_row = {
            "Timestamp": timestamp,
            "User": st.session_state.username,
            "Patient Name / ID": patient_name if patient_name else "Not provided",
            "Age": patient_age,
            "Prediction": prediction_text,
            "Benign Probability": round(benign_prob, 2),
            "Malignant Probability": round(malignant_prob, 2),
            "Confidence": round(confidence, 2)
        }
        save_prediction_history(history_row)
        st.success("Prediction saved to history.")

        report_df = pd.DataFrame([history_row])
        st.download_button("Download Result as CSV", report_df.to_csv(index=False).encode("utf-8"), "prediction_result.csv", "text/csv", use_container_width=True)

        pdf_buffer = generate_pdf_report(case_summary, prediction_text, f"{confidence:.2f}%", recommendations)
        if pdf_buffer is not None:
            st.download_button("Download PDF Report", pdf_buffer, "breast_cancer_prediction_report.pdf", "application/pdf", use_container_width=True)
        else:
            st.info("To enable PDF report download, install reportlab: pip install reportlab")
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Feature Guide Page
# =========================
elif page == "Feature Guide":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Important Input Feature Meaning")
    st.write("This guide helps users understand the major tumor measurements used in Simple Mode.")
    feature_info = {
        "Mean Radius": "Average distance from the center to the boundary points of the tumor.",
        "Mean Texture": "Variation in grayscale intensity of the tumor image.",
        "Mean Perimeter": "Average perimeter size of the tumor.",
        "Mean Area": "Average area covered by the tumor cells.",
        "Mean Smoothness": "Measures variation in radius lengths.",
        "Mean Compactness": "Indicates compactness of the tumor shape.",
        "Mean Concavity": "Measures severity of concave portions of the tumor boundary.",
        "Mean Concave Points": "Number of concave points on the tumor contour.",
        "Worst Radius": "Largest recorded radius value.",
        "Worst Concave Points": "Largest recorded concave point value."
    }
    for feature, meaning in feature_info.items():
        st.markdown(f"<div class='info-box'><b>{feature}</b><br>{meaning}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Feedback Page
# =========================
elif page == "Feedback":
    st.markdown("""
    <div class='feedback-box'>
        <h3>Share Your Feedback</h3>
        <p>
        Your feedback helps improve the Breast Cancer Prediction System. You can share your experience,
        report any issue, or suggest new features that can make the application more useful and user-friendly.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("### Feedback Form")

    fcol1, fcol2 = st.columns(2)

    with fcol1:
        feedback_name = st.text_input(
            "Name / User ID",
            value=st.session_state.username
        )

    with fcol2:
        feedback_category = st.selectbox(
            "Feedback Category",
            [
                "User Interface",
                "Prediction Experience",
                "Chatbot",
                "Report Download",
                "Feature Suggestion",
                "Other"
            ]
        )

    rating = st.radio(
        "Overall Rating",
        ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"],
        horizontal=True
    )

    rating_value = len(rating)

    feedback_text = st.text_area(
        "Additional Comments (Optional)",
        placeholder="Optional: Share suggestions, improvements, or comments...",
        height=150
    )

    if st.button("Submit Feedback", use_container_width=True):

        feedback_row = {
            "Timestamp": datetime.now().strftime("%d-%m-%Y %I:%M:%S %p"),
            "User": st.session_state.username,
            "Name / User ID": feedback_name if feedback_name else "Not provided",
            "Category": feedback_category,
            "Rating": rating_value,
            "Feedback": feedback_text.strip() if feedback_text.strip() else "No written feedback"
        }

        save_feedback(feedback_row)
        st.success("⭐ Thank you for your feedback!")

    if FEEDBACK_PATH.exists():
        with st.expander("View Recent Feedback"):
            feedback_df = pd.read_csv(FEEDBACK_PATH)
            st.dataframe(feedback_df.tail(10), use_container_width=True)

            st.download_button(
                "Download Feedback CSV",
                feedback_df.to_csv(index=False).encode("utf-8"),
                "feedback.csv",
                "text/csv",
                use_container_width=True
            )

# =========================
# About Page
# =========================
elif page == "About":
    st.markdown("""
    <div class='about-hero'>
        <h2>About Breast Cancer Prediction System</h2>
        <p>
        The Breast Cancer Prediction System is designed as a modern healthcare-focused machine learning application
        that helps analyze breast tumor measurement values and classify the tumor pattern as benign or malignant.
        Instead of presenting only a technical model output, the application provides a complete user experience with
        prediction results, risk level, confidence score, visual analysis, report generation, and chatbot assistance.
        </p>
        <p>
        The system uses tumor cell measurement features such as radius, texture, perimeter, area, smoothness,
        compactness, concavity, and concave points. These values are processed by a trained XGBoost classification
        model to generate a prediction and probability-based risk assessment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='about-section'>
        <h3>Purpose of the Application</h3>
        <p>
        This application was built to demonstrate how machine learning can be applied in the healthcare domain
        for intelligent tumor classification. It converts raw tumor measurement values into a clear prediction result
        and presents the result in a way that is understandable for users through risk labels, visual charts,
        and recommended next steps.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='about-section'>
        <h3>How the System Works</h3>
        <p>
        Users enter patient details and tumor measurement values through the prediction dashboard. The trained model
        analyzes the entered features and predicts whether the tumor pattern is more likely to be benign or malignant.
        The application then displays the confidence score, probability analysis, risk meter, case summary, and
        downloadable PDF or CSV reports.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='about-section'>
        <h3>Application Capabilities</h3>
        <p>
        The system includes secure account access, simple and advanced input modes, prediction history tracking,
        PDF report generation, CSV export, tumor feature comparison, radar chart visualization, feature explanation,
        and an interactive chatbot that helps users understand medical terms and prediction results.
        </p>
    </div>
    """, unsafe_allow_html=True)


# =========================
# History Page
# =========================
elif page == "History":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Prediction History")
    if HISTORY_PATH.exists():
        history_df = pd.read_csv(HISTORY_PATH)

        # Clean old history columns if previous app versions stored Sample ID or duplicate Patient Name columns
        if "Patient Name / ID" not in history_df.columns and "Patient Name" in history_df.columns:
            history_df["Patient Name / ID"] = history_df["Patient Name"]
        elif "Patient Name / ID" in history_df.columns and "Patient Name" in history_df.columns:
            history_df["Patient Name / ID"] = history_df["Patient Name / ID"].replace("None", pd.NA).fillna(history_df["Patient Name"])

        columns_to_remove = [col for col in ["Sample ID", "Patient Name"] if col in history_df.columns]
        if columns_to_remove:
            history_df = history_df.drop(columns=columns_to_remove)
            history_df.to_csv(HISTORY_PATH, index=False)

        final_history_columns = [
            "Timestamp", "User", "Patient Name / ID", "Age", "Prediction",
            "Benign Probability", "Malignant Probability", "Confidence"
        ]
        available_columns = [col for col in final_history_columns if col in history_df.columns]
        history_df = history_df[available_columns]

        st.dataframe(history_df.tail(50), use_container_width=True)
        st.download_button("Download Full History CSV", history_df.to_csv(index=False).encode("utf-8"), "prediction_history.csv", "text/csv", use_container_width=True)
        if st.button("Clear Prediction History", use_container_width=True):
            HISTORY_PATH.unlink(missing_ok=True)
            st.success("Prediction history cleared.")
            st.rerun()
    else:
        st.info("No predictions saved yet. Make a prediction first to create history.")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Chatbot Page
# =========================
elif page == "Chatbot":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🎗️ Breast Cancer Assistant Chatbot")
    st.write("Ask questions about tumor classification, input features, risk levels, reports, and prediction results.")
    suggested_questions = [
        "What is benign?", "What is malignant?", "What does high risk mean?",
        "What should I do if result is high risk?", "What is mean radius?",
        "What are the important features?", "What is feature importance?",
        "What is prediction history?", "What does confidence score mean?"
    ]
    selected_question = st.selectbox("Choose a quick question", ["Type my own question"] + suggested_questions)
    if selected_question == "Type my own question":
        user_question = st.text_input("Type your question here", placeholder="Example: What is mean concavity?")
    else:
        user_question = selected_question
        st.text_input("Selected question", value=user_question, disabled=True)
    if st.button("Ask Assistant", use_container_width=True):
        if user_question.strip() != "":
            answer = chatbot_response(user_question)
            st.session_state.chat_history.append((user_question, answer))
        else:
            st.warning("Please type a question or choose one from the list.")
    if st.session_state.chat_history:
        st.write("### Chat History")
        for question, answer in reversed(st.session_state.chat_history[-7:]):
            st.markdown(f"<div class='chat-question'>You: {question}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-answer'><b>Assistant:</b><br>{answer}</div>", unsafe_allow_html=True)
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
