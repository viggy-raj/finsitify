import streamlit as st
import subprocess
import os
import pandas as pd
from datetime import datetime
from pathlib import Path

# --- Configuration ---
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
MANIFEST_FILE = Path("data") / "manifest.csv"

# --- Utility Functions ---

def run_script(script_name: str, args: list = []):
    """Execute a Python script using subprocess and capture output."""
    command = ["python", script_name] + args
    st.info(f"Executing: {' '.join(command)}")
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        st.success(f"✅ {script_name} execution successful!")
        if process.stdout:
            st.code(process.stdout)
        if process.stderr:
            st.warning(f"Script stderr (often warnings): {process.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"❌ {script_name} failed!")
        st.code(f"Error Code: {e.returncode}\n\nSTDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}")
        return False
    except FileNotFoundError:
        st.error(f"❌ Could not find {script_name}. Ensure it's in the current directory.")
        return False

def display_file_content(file_path: Path):
    """Display the content of a CSV file if it exists."""
    if file_path.exists():
        st.subheader(f"Content of {file_path.name}")
        try:
            df = pd.read_csv(file_path)
            st.dataframe(df)
        except Exception as e:
            st.warning(f"Could not load or display CSV: {e}")
    else:
        st.warning(f"File not found: {file_path}")

# --- Streamlit App Layout ---

st.set_page_config(
    page_title="Finsitify ML Pipeline",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Finsitify: Market Prediction Pipeline")
st.markdown("A unified interface to manage the Data, Model, Sentiment, and Prediction stages.")

# Create tabs for navigation
tab_data, tab_sentiment, tab_model, tab_combine = st.tabs([
    "1. Data Ingestion", 
    "2. Sentiment Analysis", 
    "3. Base Model Training", 
    "4. Final Prediction & Evaluation"
])

# ==============================================================================
# 1. Data Ingestion Tab (data.py)
# ==============================================================================
with tab_data:
    st.header("Step 1: Data Ingestion (`data.py`)")
    st.markdown("Fetch public financial, climate, and environmental datasets.")

    with st.form("data_form"):
        # Default dates covering a long history up to near-present
        default_start = "2015-01-01"
        default_end = datetime.now().strftime('%Y-%m-%d')
        start_date = st.text_input("Start Date (YYYY-MM-DD)", default_start)
        end_date = st.text_input("End Date (YYYY-MM-DD)", default_end)
        
        submitted = st.form_submit_button("🚀 Run Data Ingestion")
        
        if submitted:
            run_script("data.py", ["--start", start_date, "--end", end_date])
    
    st.subheader("Data Manifest & Status")
    if MANIFEST_FILE.exists():
        display_file_content(MANIFEST_FILE)
    else:
        st.info("Run the Data Ingestion step to generate the data/manifest.csv.")

# ==============================================================================
# 2. Sentiment Analysis Tab (sentimental.py)
# ==============================================================================
with tab_sentiment:
    st.header("Step 2: Sentiment Analysis (`sentimental.py`)")
    st.markdown("Scrape financial news and run the FinBERT model to get the latest market sentiment.")

    if st.button("🤖 Run Sentiment Analysis"):
        run_script("sentimental.py")

    sentiment_summary_file = OUTPUT_DIR / "sentiment_summary.csv"
    st.subheader("Latest Sentiment Summary")
    display_file_content(sentiment_summary_file)
    
    st.caption("Sentiment details are saved to outputs/sentiment_details.csv")

# ==============================================================================
# 3. Base Model Training Tab (xgboost_model.py)
# ==============================================================================
with tab_model:
    st.header("Step 3: Base Model Training (`xgboost_model.py`)")
    st.markdown("Train the initial XGBoost model on all ingested data using the specified target column.")

    with st.form("model_form"):
        target_col = st.text_input("Target Column Name", "close")
        
        submitted = st.form_submit_button("🧠 Train Base XGBoost Model")
        
        if submitted:
            run_script("xgboost_model.py", ["--target", target_col])
    
    st.subheader("Base Model Predictions")
    base_pred_file = OUTPUT_DIR / "xgboost_predictions.csv"
    display_file_content(base_pred_file)
    st.caption("The model and scaler files are saved in the 'models' directory.")

# ==============================================================================
# 4. Final Prediction & Evaluation Tab (predictions.py)
# ==============================================================================
with tab_combine:
    st.header("Step 4: Combine & Evaluate (`predictions.py`)")
    st.markdown("Combine base predictions with sentiment, train a meta-model, and project the final result.")

    with st.form("combine_form"):
        future_steps = st.number_input(
            "Future Projection Steps (N)", 
            min_value=0, 
            max_value=100, 
            value=30, 
            step=1,
            help="Project N future steps using linear trend for visualization."
        )
        retrain_combiner = st.checkbox("Force Retrain Combiner Model", False, help="Force retrain of the combination model, even if one exists.")

        args = ["--future", str(future_steps), "--plot"]
        if retrain_combiner:
            args.append("--retrain")
            
        submitted = st.form_submit_button("📊 Generate Final Verdict")

        if submitted:
            # We cannot display the plot directly in a subprocess, but the terminal
            # output with the final metrics will be captured.
            st.warning("The plot from predictions.py will not display directly here (it runs in a separate process), but the final metrics are captured below.")
            run_script("predictions.py", args)
            
    st.subheader("Combined Predictions")
    combo_pred_file = OUTPUT_DIR / "combined_predictions.csv"
    display_file_content(combo_pred_file)
    st.caption("The combination model is saved as models/combination_xgb.pkl.")
