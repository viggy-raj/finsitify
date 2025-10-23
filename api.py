import os
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import io
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.metrics import mean_squared_error

# --- Configuration ---
OUTPUT_DIR = Path("outputs")
DATA_DIR = Path("data")
MANIFEST_FILE = DATA_DIR / "manifest.csv"
COMBO_PRED_FILE = OUTPUT_DIR / "combined_predictions.csv"
SENTIMENT_SUMMARY_FILE = OUTPUT_DIR / "sentiment_summary.csv"

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

app = FastAPI(title="Finsitify ML Prediction API", version="1.0.0")

# --- Schemas for API Response ---
class FinalVerdict(BaseModel):
    """Schema for the final prediction and metrics."""
    timestamp: datetime
    latest_prediction: float
    base_prediction: float
    true_value: Optional[float] = None
    sentiment_score: float
    sentiment_label: str
    combined_rmse: float
    combined_accuracy_pct: float
    combined_confidence_pct: float

class TimeseriesData(BaseModel):
    """Schema for timeseries data, represented as a list of dicts."""
    data: list

class PipelineStatus(BaseModel):
    """Schema for the pipeline execution status."""
    success: bool
    log_output: str

class DataUpload(BaseModel):
    """Schema for receiving custom data and pipeline configuration."""
    # Custom data as a raw CSV string
    csv_content: str
    target_column: str = 'close'
    # Filename to save the custom data as in the data/ folder
    filename: str = 'custom_input.csv'

# --- Utility Functions ---

def run_script(script_name: str, args: list = []) -> str:
    """Execute a Python script and return stdout/stderr."""
    command = ["python", script_name] + args
    logging.info(f"Executing: {' '.join(command)}")
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=600 # 10 minutes timeout for the whole pipeline
        )
        log = f"✅ SUCCESS: {script_name}\nSTDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
        logging.info(log)
        return log
    except subprocess.CalledProcessError as e:
        log = f"❌ FAILURE: {script_name}\nError Code: {e.returncode}\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
        logging.error(log)
        raise RuntimeError(log) from e
    except subprocess.TimeoutExpired as e:
        log = f"❌ FAILURE: {script_name} timed out."
        logging.error(log)
        raise RuntimeError(log) from e
    except FileNotFoundError:
        log = f"❌ FAILURE: Could not find {script_name}."
        logging.error(log)
        raise RuntimeError(log)

def compute_metrics(y_true, y_pred):
    """Re-implementation of the compute_metrics from predictions.py"""
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mean_val = np.mean(np.abs(y_true)) if np.mean(np.abs(y_true)) != 0 else 1.0
    accuracy_pct = max(0.0, 100.0 * (1 - rmse / mean_val))
    residuals = np.abs(y_true - y_pred)
    conf = max(0.0, 100.0 * (1 - np.var(residuals) / (np.var(y_true) + 1e-9)))
    return rmse, accuracy_pct, conf

def save_custom_csv(filename: str, csv_content: str):
    """Saves the CSV content string received via API to the data directory."""
    path = DATA_DIR / filename
    try:
        # Use StringIO to read the string content as if it were a file
        df = pd.read_csv(io.StringIO(csv_content))
        # Save the DataFrame to a file in the data/ folder
        df.to_csv(path, index=False)
        logging.info(f"Saved custom data to {path}")
    except Exception as e:
        logging.error(f"Failed to process/save custom CSV: {e}")
        # Reraise as an HTTP error for the API consumer
        raise HTTPException(status_code=400, detail=f"Invalid CSV content provided: {e}")

# --- API Endpoints ---

@app.post("/pipeline/run_with_custom_data", response_model=PipelineStatus)
def run_pipeline_with_custom_data(upload: DataUpload):
    """
    Triggers the pipeline using custom data provided in the payload.
    The custom data will be added to the data/ folder and included in model training.
    """
    full_log = ""
    try:
        # 1. Save Custom Data
        save_custom_csv(upload.filename, upload.csv_content)
        full_log += f"Saved custom data: {upload.filename} successfully.\n"
        
        # 2. Sentiment Analysis (Runs on live news)
        log_sent = run_script("sentimental.py")
        full_log += log_sent + "\n"
        
        # 3. Base Model Training (Will load ALL CSVs in data/ including custom_input.csv)
        log_xgb = run_script("xgboost_model.py", ["--target", upload.target_column])
        full_log += log_xgb + "\n"
        
        # 4. Final Prediction & Combine 
        log_pred = run_script("predictions.py", ["--future", "30", "--retrain"])
        full_log += log_pred + "\n"
        
        return PipelineStatus(success=True, log_output=full_log)
    except Exception as e:
        # If any script fails, return a failure status with the log up to the point of failure
        return PipelineStatus(success=False, log_output=full_log + f"\n[CRITICAL FAILURE]: {e}")


@app.post("/pipeline/run", response_model=PipelineStatus)
def run_full_pipeline(target_col: str = 'close', start_date: str = '2015-01-01', end_date: str = datetime.now().strftime('%Y-%m-%d')):
    """Triggers a full run of the data ingestion (from public sources), sentiment, model training, and prediction pipeline."""
    full_log = ""
    try:
        # 1. Data Ingestion (Uses external public data sources)
        log_data = run_script("data.py", ["--start", start_date, "--end", end_date])
        full_log += log_data + "\n"
        
        # 2. Sentiment Analysis
        log_sent = run_script("sentimental.py")
        full_log += log_sent + "\n"
        
        # 3. Base Model Training
        log_xgb = run_script("xgboost_model.py", ["--target", target_col])
        full_log += log_xgb + "\n"
        
        # 4. Final Prediction & Combine (Explicitly run without plotting on the server)
        log_pred = run_script("predictions.py", ["--future", "30", "--retrain"])
        full_log += log_pred + "\n"
        
        return PipelineStatus(success=True, log_output=full_log)
    except Exception as e:
        return PipelineStatus(success=False, log_output=full_log + f"\n[CRITICAL FAILURE]: {e}")

@app.get("/data/manifest", response_model=TimeseriesData)
def get_data_manifest():
    """Returns the content of the data/manifest.csv file."""
    if not MANIFEST_FILE.exists():
        raise HTTPException(status_code=404, detail="Data manifest not found. Run the pipeline first.")
    try:
        df = pd.read_csv(MANIFEST_FILE)
        return TimeseriesData(data=df.to_dict(orient="records"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading manifest file: {e}")

@app.get("/prediction/combined", response_model=TimeseriesData)
def get_combined_predictions_timeseries():
    """Returns the full timeseries of combined predictions (y_true, y_pred, combined_pred, sentiment_score)."""
    if not COMBO_PRED_FILE.exists():
        raise HTTPException(status_code=404, detail="Combined predictions not found. Run the pipeline first.")
    try:
        df = pd.read_csv(COMBO_PRED_FILE)
        return TimeseriesData(data=df.to_dict(orient="records"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading combined predictions file: {e}")

@app.get("/prediction/verdict", response_model=FinalVerdict)
def get_final_verdict():
    """Returns the final, single-point market verdict (latest combined prediction and metrics)."""
    if not COMBO_PRED_FILE.exists():
        raise HTTPException(status_code=404, detail="Combined predictions not found. Run the pipeline first.")
    if not SENTIMENT_SUMMARY_FILE.exists():
        raise HTTPException(status_code=404, detail="Sentiment summary not found. Run the pipeline first.")
        
    try:
        df_pred = pd.read_csv(COMBO_PRED_FILE)
        df_sent = pd.read_csv(SENTIMENT_SUMMARY_FILE)
        
        if df_pred.empty or df_sent.empty:
             raise HTTPException(status_code=404, detail="Prediction or sentiment data is empty.")

        # Latest combined prediction
        latest_pred_row = df_pred.iloc[-1]
        
        # Calculate full metrics
        rmse, acc_pct, conf = compute_metrics(df_pred['y_true'].values, df_pred['combined_pred'].values)
        
        # Latest sentiment
        latest_sent_row = df_sent.iloc[-1]
        
        # Handle cases where true_value might not be available (e.g., in a pure forecast)
        true_val = float(latest_pred_row['y_true']) if 'y_true' in latest_pred_row and not pd.isna(latest_pred_row['y_true']) else None

        return FinalVerdict(
            timestamp=datetime.utcnow(),
            latest_prediction=float(latest_pred_row['combined_pred']),
            base_prediction=float(latest_pred_row['y_pred']),
            true_value=true_val,
            sentiment_score=float(latest_sent_row['sentiment_score']),
            sentiment_label=str(latest_sent_row['sentiment_label']),
            combined_rmse=float(rmse),
            combined_accuracy_pct=float(acc_pct),
            combined_confidence_pct=float(conf)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing final verdict: {e}")

# If you use the API to start, you'll need the following block (as in the original code)
if __name__ == "__main__":
    import uvicorn
    # Initial run to populate files (optional, but good for first boot)
    try:
        run_full_pipeline(target_col='close')
    except Exception as e:
        logging.error(f"Initial pipeline run failed: {e}")
    
    # Use a fixed port 8000 for standard server deployments
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))