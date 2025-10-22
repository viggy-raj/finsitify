#!/bin/bash

# --- 1. Initial Setup and Data Fetching (Pre-API Start) ---
# Ensure directories exist for the first run
mkdir -p data
mkdir -p outputs
mkdir -p models

# Run the full pipeline once to generate all initial data/model files.
# This might take a while, but it's crucial for the API endpoints to work.
echo "--- Running Initial Data Pipeline (Step 1-4) ---"
python data.py --start 2015-01-01 --end $(date +%Y-%m-%d)
python sentimental.py
python xgboost_model.py --target close
# IMPORTANT: Removed the '--plot' flag to avoid dependency on a display server.
python predictions.py --future 30 --retrain

# --- 2. Start the API Service ---
echo "--- Starting FastAPI Server ---"
# Standardized to fixed port 8000 for server deployments (like AWS ECS/K8s)
# The deployment environment should be configured to route traffic to this port.
uvicorn api:app --host 0.0.0.0 --port 8000
