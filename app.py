from pathlib import Path
import pandas as pd
import numpy as np
import re
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Bengaluru_House_Data.csv"

# Page Configuration
st.set_page_config(
    page_title="Bengaluru House Price Prediction | Enterprise AI", 
    page_icon="", 
    layout="centered"
)

# Custom Modern Enterprise SaaS CSS Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 5px;
    }
    div[data-testid="stForm"] {
        background: linear-gradient(145deg, #111827 0%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 35px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
    }
    .form-section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 20px;
        letter-spacing: -0.01em;
        border-left: 3px solid #6366f1;
        padding-left: 10px;
    }
    label p {
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: #9ca3af !important;
    }
    .stSelectbox div[data-baseweb="select"], .stNumberInput input {
        background-color: #1a2234 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    .stButton button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 0.75rem 1.5rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
        box-shadow: 0 6px 16px rgba(99, 102, 241, 0.5);
    }
    .team-banner {
        background: rgba(17, 24, 39, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-left: 4px solid #6366f1;
        padding: 20px 24px;
        border-radius: 0 12px 12px 0;
        margin-bottom: 30px;
        font-size: 0.875rem;
        color: #9ca3af;
    }
    .team-title {
        color: #e5e7eb;
        font-weight: 700;
        margin-bottom: 10px;
        font-size: 0.95rem;
    }
    .team-member {
        display: flex;
        justify-content: space-between;
        padding: 4px 0;
        border-bottom: 1px dashed rgba(255, 255, 255, 0.05);
        max-width: 450px;
    }
    .member-name {
        color: #f3f4f6;
        font-weight: 500;
    }
    .member-uid {
        color: #818cf8;
        font-family: monospace;
    }
    .team-section {
        margin-top: 12px;
        color: #9ca3af;
        font-size: 0.85rem;
    }
    .team-section b {
        color: #e5e7eb;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff;
        font-weight: 800;
        font-size: 2.2rem;
    }
    .result-card {
        background: linear-gradient(145deg, #111827 0%, #172033 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 24px;
        margin-top: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .insight-card {
        background: linear-gradient(145deg, #111827 0%, #131d30 100%);
        border: 1px solid rgba(129, 140, 248, 0.22);
        border-radius: 12px;
        padding: 16px 18px;
        min-height: 105px;
    }
    .insight-label {
        color: #9ca3af;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
    }
    .insight-value {
        color: #f9fafb;
        font-size: 1.55rem;
        font-weight: 800;
    }
    .insight-note {
        color: #9ca3af;
        font-size: 0.78rem;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Team Information Banner
st.markdown("""
    <div class="team-banner">
        <div class="team-title">Project Created by:</div>
        <div class="team-member">
            <span class="member-name">Samrat Tripathi</span>
            <span class="member-uid">UID: 26BAI70082</span>
        </div>
        <div class="team-member">
            <span class="member-name">Chandan Sharma</span>
            <span class="member-uid">UID: 26BAI70060</span>
        </div>
        <div class="team-member">
            <span class="member-name">Anshu Yadav</span>
            <span class="member-uid">UID: 26BAI70062</span>
        </div>
        <div class="team-section">
            <b>Section:</b> 26AML-101B
        </div>
    </div>
""", unsafe_allow_html=True)

st.title("House Price Prediction")
st.markdown("<p style='color: #9ca3af; margin-bottom: 25px;'>Explore data-driven residential price estimates, model evaluation metrics, and the machine-learning workflow behind each prediction.</p>", unsafe_allow_html=True)

@st.cache_resource
def get_trained_model_and_locations():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Bengaluru_House_Data.csv not found in the project directory.")
    
    data = pd.read_csv(DATA_PATH)
    df = data.copy()
    df = df.dropna(subset=["location", "size", "total_sqft", "bath", "price"])
    
    def parse_sqft(value):
        if pd.isna(value):
            return np.nan
        text = str(value).strip().replace(",", "")
        numbers = re.findall(r"\d+(?:\.\d+)?", text)
        if not numbers:
            return np.nan
        vals = [float(n) for n in numbers]
        return sum(vals) / len(vals) if len(vals) > 1 else vals[0]

    df["total_sqft"] = df["total_sqft"].apply(parse_sqft)
    df["BHK"] = df["size"].astype(str).str.extract(r"(\d+)")[0]
    df["BHK"] = pd.to_numeric(df["BHK"], errors="coerce")
    df["bath"] = pd.to_numeric(df["bath"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df.dropna(subset=["location", "total_sqft", "BHK", "bath", "price"])
    df = df[(df["total_sqft"] > 0) & (df["BHK"] > 0) & (df["bath"] > 0) & (df["price"] > 0)]
    df["location"] = df["location"].astype(str).str.strip()
    df = df[(df["total_sqft"] / df["BHK"] >= 150) & (df["total_sqft"] / df["BHK"] <= 3000)]
    df = df[df["bath"] <= df["BHK"] + 2]

    # Train a base model on physical property attributes only. Location is
    # applied as a separate, data-derived calibration so its effect is explicit.
    feature_columns = ["total_sqft", "bath", "BHK"]
    X = df[feature_columns].copy()
    y = df["price"].copy()
    locations_all = df["location"].copy()

    X_train, X_test, y_train, y_test, loc_train, loc_test = train_test_split(
        X, y, locations_all, test_size=0.2, random_state=42
    )

    # Compute location median price/sq.ft. from TRAINING data only to avoid
    # leaking test labels into model evaluation.
    train_ppsf = y_train / X_train["total_sqft"]
    location_ppsf = train_ppsf.groupby(loc_train).median().to_dict()
    global_ppsf = float(train_ppsf.median())

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, ["total_sqft", "bath", "BHK"])
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1, min_samples_leaf=2
        ))
    ])
    model.fit(X_train, y_train)

    # Apply location calibration to held-out predictions, just as at inference.
    base_test_predictions = model.predict(X_test)
    test_location_ppsf = loc_test.map(location_ppsf).fillna(global_ppsf)
    location_factors = (test_location_ppsf / global_ppsf).clip(0.5, 2.0)
    test_predictions = base_test_predictions * location_factors.to_numpy()
    metrics = {
        "mae": float(mean_absolute_error(y_test, test_predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, test_predictions))),
        "r2": float(r2_score(y_test, test_predictions)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "clean_rows": int(len(df)),
    }

    locations = sorted(df["location"].dropna().astype(str).unique().tolist())

    return model, locations, metrics, location_ppsf, global_ppsf

try:
    model, locations, metrics, location_ppsf, global_ppsf = get_trained_model_and_locations()
except Exception as exc:
    st.error(f"Initialization failed: {exc}")
    st.stop()

# Visible model evaluation panel: metrics are calculated from the held-out test split.
st.markdown("### Model performance")
st.caption("Measured on a held-out 20% test split from the cleaned dataset. Errors are expressed in ₹ lakh.")
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(
        f'<div class="insight-card"><div class="insight-label">Mean Absolute Error</div>'
        f'<div class="insight-value">₹{metrics["mae"]:.2f}L</div>'
        f'<div class="insight-note">Lower is better</div></div>',
        unsafe_allow_html=True
    )
with m2:
    st.markdown(
        f'<div class="insight-card"><div class="insight-label">Root Mean Squared Error</div>'
        f'<div class="insight-value">₹{metrics["rmse"]:.2f}L</div>'
        f'<div class="insight-note">Penalizes large errors</div></div>',
        unsafe_allow_html=True
    )
with m3:
    st.markdown(
        f'<div class="insight-card"><div class="insight-label">R² Score</div>'
        f'<div class="insight-value">{metrics["r2"]:.3f}</div>'
        f'<div class="insight-note">Higher is generally better</div></div>',
        unsafe_allow_html=True
    )

with st.expander("How the prediction works", expanded=False):
    st.markdown(
        "**Algorithm:** Random Forest Regressor with 100 decision trees.  \\n"
        "**Input features:** location, total area, bathrooms, and BHK.  \\n"
        "**Preprocessing:** missing-value imputation, one-hot encoding for location, and numerical feature preparation.  \\n"
        "**Evaluation:** a reproducible 80/20 train-test split (random state 42).  \\n"
        "**Dataset size after cleaning:** "
        f"{metrics['clean_rows']:,} records ({metrics['train_rows']:,} training / {metrics['test_rows']:,} testing)."
    )

# Input Form
with st.form("prediction_form"):
    st.markdown('<div class="form-section-title">Property Parameters</div>', unsafe_allow_html=True)
    st.caption("Change the property details, then click Estimate Property Price to refresh the valuation. Location adjusts the base estimate using historical median price per sq. ft. from training data.")
    
    location = st.selectbox("Location", locations)
    total_sqft = st.number_input("Total Area (sq. ft.)", min_value=150.0, max_value=100000.0, value=1000.0, step=50.0)
    
    col1, col2 = st.columns(2)
    with col1:
        bath = st.number_input("Bathrooms", min_value=1, max_value=30, value=2, step=1)
    with col2:
        bhk = st.number_input("Bedrooms (BHK)", min_value=1, max_value=50, value=2, step=1)
        
    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Estimate Property Price", use_container_width=True)

if submitted:
    if total_sqft / bhk < 150:
        st.warning("Notice: The area is unusually small for this number of bedrooms; prediction accuracy may vary.")
    if bath > bhk + 2:
        st.warning("Notice: This bathroom-to-bedroom combination is uncommon in historical training records.")
        
    features = pd.DataFrame(
        [[float(total_sqft), float(bath), float(bhk)]],
        columns=["total_sqft", "bath", "BHK"],
    )

    try:
        # Calibrate the base estimate by the selected location's median
        # training price/sq.ft. relative to the global training median.
        base_prediction = float(model.predict(features)[0])
        selected_ppsf = float(location_ppsf.get(location, global_ppsf))
        location_factor = float(np.clip(selected_ppsf / global_ppsf, 0.5, 2.0))
        predicted_lakh = base_prediction * location_factor
        if not pd.notna(predicted_lakh) or predicted_lakh <= 0:
            st.error("The model returned an invalid valuation. Check the dataset configuration.")
        else:
            st.markdown(f"""
                <div class="result-card">
                    <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #818cf8; font-weight: 600; margin-bottom: 4px;">Valuation Result</div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #ffffff;">₹{predicted_lakh:,.2f} <span style="font-size: 1.2rem; font-weight: 500; color: #9ca3af;">lakh</span></div>
                    <div style="font-size: 0.85rem; color: #9ca3af; margin-top: 6px;">Approximate Equivalent: ₹{predicted_lakh * 100000:,.0f}</div>
                </div>
            """, unsafe_allow_html=True)
    except Exception as exc:
        st.error(f"Prediction execution failed: {exc}")