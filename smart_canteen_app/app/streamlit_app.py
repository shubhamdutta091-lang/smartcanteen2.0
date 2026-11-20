# SmartCanteen: ML-Based Food Demand and Waste Reduction System
# Main Streamlit Application - Fixed Version

import os
import sys
import joblib
import datetime
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure the project root is on the import path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -------------------------------------------------------------------
# Streamlit configuration (Must be the first Streamlit command)
# Note: 'browser.gatherUsageStats' must be set in .streamlit/config.toml,
# not via st.set_option(), which raises a StreamlitAPIException.
# -------------------------------------------------------------------
st.set_page_config(
    page_title="SmartCanteen | ML Food Demand & Waste Reduction",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------
# Simple placeholder authentication
# -------------------------------------------------------------------
def authenticate():
    """Very simple username/password check. Replace with real auth as needed."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.sidebar.subheader("🔑 User Login")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if username == "admin" and password == "admin":
                st.session_state.authenticated = True
                st.rerun()  # Immediately refresh UI to hide login form
            else:
                st.sidebar.error("Invalid credentials")
    return st.session_state.authenticated

# Initialise authentication – stop the app for unauthenticated users
if not authenticate():
    st.info("👈 Please log in from the sidebar to access SmartCanteen (Default: `admin` / `admin`).")
    st.stop()

# -------------------------------------------------------------------
# Custom CSS
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"] {font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;}
        .hero-card {background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 55%, #388E3C 100%); color:#fff; padding:30px 32px; border-radius:18px; box-shadow:0 10px 25px rgba(46,125,50,0.18); margin-bottom:28px;}
        .hero-badge {display:inline-block; background:rgba(255,255,255,0.2); padding:4px 12px; border-radius:20px; font-size:0.78rem; font-weight:700; margin-bottom:10px;}
        .hero-card h1 {font-size:2.3rem; font-weight:800; margin:0 0 8px 0;}
        .hero-card p {font-size:1.08rem; margin:0;}
        .metric-card-box {background:#fff; border:1px solid #E2E8F0; border-radius:14px; padding:20px 22px; box-shadow:0 2px 10px rgba(0,0,0,0.04); margin-bottom:16px;}
        .metric-card-box:hover {transform:translateY(-2px); box-shadow:0 6px 18px rgba(46,125,50,0.10); border-color:#A5D6A7;}
        .footer-container {text-align:center; padding:24px 10px; margin-top:55px; border-top:1px solid #E2E8F0; color:#2E7D32; font-weight:600; font-size:0.95rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Import project modules
# -------------------------------------------------------------------
from src.data_generator import generate_canteen_dataset, FOOD_CATALOG
from src.data_preprocessing import clean_canteen_data, validate_dataset_columns
from src.feature_engineering import create_features
from src.train_model import train_and_evaluate_models
from src.predict import load_trained_model, predict_demand
from src.waste_calculator import calculate_waste_metrics

# -------------------------------------------------------------------
# Helper paths & directory initialization
# -------------------------------------------------------------------
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "sample_canteen_data.csv")
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "food_demand_model.pkl")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "prediction_report.csv")

for path in [DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH, REPORT_PATH]:
    os.makedirs(os.path.dirname(path), exist_ok=True)

# -------------------------------------------------------------------
# Caching helpers
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_cached_sample_data():
    """Load or generate the default dataset."""
    if os.path.exists(DEFAULT_DATA_PATH):
        df = pd.read_csv(DEFAULT_DATA_PATH)
    else:
        df = generate_canteen_dataset(output_path=DEFAULT_DATA_PATH)
    return clean_canteen_data(df)

def initialize_session():
    if "current_df" not in st.session_state:
        st.session_state.current_df = get_cached_sample_data()
    if "model_bundle" not in st.session_state:
        if os.path.exists(DEFAULT_MODEL_PATH):
            try:
                loaded = load_trained_model(DEFAULT_MODEL_PATH)
                # Ensure loaded object matches expected dictionary bundle structure
                if isinstance(loaded, dict):
                    st.session_state.model_bundle = loaded
                else:
                    st.session_state.model_bundle = {
                        "best_model": loaded,
                        "best_model_name": type(loaded).__name__,
                    }
            except Exception:
                st.session_state.model_bundle = None
        else:
            st.session_state.model_bundle = None
    if "predictions_history" not in st.session_state:
        st.session_state.predictions_history = []

initialize_session()

# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
def render_footer():
    st.markdown(
        """
        <div class="footer-container">
            🌱 SmartCanteen | Reducing Food Waste with Machine Learning
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------------
# Sidebar navigation
# -------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="text-align:center; padding:14px 0 8px 0;">
        <h2 style="color:#2E7D32; font-weight:800; margin:0; font-size:1.7rem;">🥗 SmartCanteen</h2>
        <span style="background:#E8F5E9; color:#2E7D32; font-size:0.78rem; padding:4px 12px; border-radius:12px; font-weight:700; text-transform:uppercase;">ML Demand Planner</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

PAGES = ["Home", "Upload/View Data", "Data Analysis", "Train ML Model", "Predict Demand", "Waste Report", "About"]
page_icons = {
    "Home": "🏠",
    "Upload/View Data": "📂",
    "Data Analysis": "📊",
    "Train ML Model": "🤖",
    "Predict Demand": "🔮",
    "Waste Report": "📋",
    "About": "ℹ️",
}
selected_page = st.sidebar.radio("Navigation Menu", PAGES, format_func=lambda x: f"{page_icons[x]}  {x}")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Canteen Control Center")

if st.session_state.current_df is not None:
    st.sidebar.success(f"📊 Dataset: **{len(st.session_state.current_df):,} records**")
else:
    st.sidebar.warning("📊 Dataset: Not Loaded")

if st.session_state.model_bundle is not None:
    best_name = (
        st.session_state.model_bundle.get("best_model_name", "Trained Model")
        if isinstance(st.session_state.model_bundle, dict)
        else type(st.session_state.model_bundle).__name__
    )
    st.sidebar.info(f"🤖 Active Model: **{best_name}**")
else:
    st.sidebar.error("🤖 Model: Not Trained Yet")

st.sidebar.caption("v2.0 • Sustainable Smart Campus AI")

# -------------------------------------------------------------------
# Page implementations
# -------------------------------------------------------------------
if selected_page == "Home":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">AI FOR CAMPUS SUSTAINABILITY</div>
            <h1>SmartCanteen: ML-Based Food Demand & Waste Reduction</h1>
            <p>A smart machine learning application empowering college canteens to forecast daily food consumption, optimise kitchen batching with automated safety buffers, and eliminate food waste.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("🎯 Objective & Problem Statement")
    st.write(
        """Every day, institutional college canteens face huge demand fluctuations driven by exam timetables, weekday attendance, weather, and campus events. Without data-driven planning, kitchen staff rely on intuition, leading to two critical bottlenecks:\n- **Massive Surplus Food Waste** – 15-35% of prepared meals are discarded.\n- **Costly Meal Shortages** – Students encounter stock-outs during peak lunch hours.\nSmartCanteen resolves this by training supervised ML models on historical sales, calendar cues, attendance, and weather to forecast next-day demand with a 10% safety stock buffer."""
    )
    render_footer()

elif selected_page == "Upload/View Data":
    st.header("📂 Dataset Management & Inspection")
    upload_col1, upload_col2 = st.columns([1.5, 1])
    with upload_col1:
        uploaded_file = st.file_uploader("📤 Upload Canteen CSV File (Optional)", type=["csv"])
    with upload_col2:
        if st.button("🔄 Generate Fresh 365-Day Synthetic Dataset", use_container_width=True):
            with st.spinner("Generating synthetic dataset…"):
                fresh_df = generate_canteen_dataset(output_path=DEFAULT_DATA_PATH)
                st.session_state.current_df = clean_canteen_data(fresh_df)
                st.cache_data.clear()
                st.success("New dataset generated!")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            validate_dataset_columns(df)
            df = clean_canteen_data(df)
            st.session_state.current_df = df
            st.success("Uploaded data validated, cleaned, and loaded.")
        except Exception as e:
            st.error(f"Error processing uploaded CSV: {e}")

    if st.session_state.current_df is not None:
        st.dataframe(st.session_state.current_df.head(20), use_container_width=True)
    render_footer()

elif selected_page == "Data Analysis":
    st.header("📊 Data Analysis")
    df = st.session_state.current_df
    if df is not None:
        st.subheader("Raw Data Overview")
        st.dataframe(df.describe(), use_container_width=True)
        if "Date" in df.columns and "Quantity_Prepared" in df.columns:
            st.subheader("Time Series of Quantity Prepared")
            fig = px.line(df, x="Date", y="Quantity_Prepared", title="Prepared Quantity Over Time")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Required columns ('Date', 'Quantity_Prepared') not found for plotting.")
    else:
        st.warning("No dataset loaded.")
    render_footer()

elif selected_page == "Train ML Model":
    st.header("🤖 Train ML Model")
    if st.session_state.current_df is None:
        st.warning("Please load or generate a dataset first.")
    elif st.button("Start Training", use_container_width=True):
        with st.spinner("Training models…"):
            try:
                model_bundle = train_and_evaluate_models(st.session_state.current_df)
                st.session_state.model_bundle = model_bundle
                # Save the full model_bundle dictionary so metadata (e.g. best_model_name,
                # feature columns, encoders) persists across sessions
                joblib.dump(model_bundle, DEFAULT_MODEL_PATH)
                st.success("Model training complete and saved!")
            except Exception as e:
                st.error(f"Training failed: {e}")
    render_footer()

elif selected_page == "Predict Demand":
    st.header("🔮 Predict Tomorrow's Demand")
    if st.session_state.model_bundle is None:
        st.error("No trained model available. Train a model first.")
    else:
        if st.button("Run Prediction", use_container_width=True):
            with st.spinner("Predicting…"):
                try:
                    prediction = predict_demand(
                        st.session_state.model_bundle, st.session_state.current_df
                    )
                    st.session_state.predictions_history.append(prediction)

                    # Safely handle scalar, array, or DataFrame return values
                    if isinstance(prediction, (int, float)):
                        st.success(f"Predicted demand: {prediction:.2f} units")
                    elif isinstance(prediction, pd.DataFrame):
                        st.success("Predictions generated:")
                        st.dataframe(prediction, use_container_width=True)
                    else:
                        st.success(f"Predicted demand: {prediction}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
    render_footer()

elif selected_page == "Waste Report":
    st.header("📋 Waste Report")
    if st.session_state.model_bundle is None:
        st.error("Model not trained – cannot compute waste metrics.")
    else:
        try:
            waste_df = calculate_waste_metrics(
                st.session_state.current_df, st.session_state.model_bundle
            )
            st.dataframe(waste_df.head(20), use_container_width=True)
            csv = waste_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Report",
                data=csv,
                file_name="waste_report.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Could not generate waste report: {e}")
    render_footer()

elif selected_page == "About":
    st.header("ℹ️ About SmartCanteen")
    st.write(
        """SmartCanteen demonstrates how machine learning can optimise food provisioning in institutional settings, reducing waste and improving student satisfaction.\n\n*Version 2.0 – Sustainable Smart Campus AI*"""
    )
    render_footer()