#!/bin/bash

# --- 1. Initial Setup and Data Fetching (Pre-API Start) ---
# Ensure directories exist for the first run
mkdir -p data
mkdir -p outputs
mkdir -p models

# Run the full pipeline once to generate all initial data/model files.
# This might take a while, but it's crucial for the API endpoints to work.
echo "--- Running Initial Data Pipeline (Step 1-4) ---"
#python data.py --start 2015-01-01 --end $(date +%Y-%m-%d)
#python sentimental.py
#python xgboost_model.py --target close
# Use --noplot since there is no display server on Render
#python predictions.py --future 30 --retrain --plot

 python -m venv .venv
 .venv\Scripts\Activate.ps1
 pip install -r requirements.txt

 streamlit run index.py


# --- 2. Start the API Service ---
echo "--- Starting FastAPI Server ---"
# $PORT is an environment variable set automatically by Render
# api:app refers to the FastAPI object 'app' in the file 'api.py'

uvicorn api:app --host 0.0.0.0 --port $PORT
