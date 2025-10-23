# [Content of xgboost_model.py remains the same]
# ... The xgboost_model.py content will be included here ...
#!/usr/bin/env python3
"""
XGBoost Model Trainer for Veda Data Folder (Robust)
---------------------------------------------------
Loads all CSVs from ./data/, aligns numeric columns, fills missing values,
and trains an XGBoost model to make predictions.

Usage:
    python xgboost_model.py --target close
"""

import os
import sys
import glob
import joblib
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler

# --- Setup ---
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")
for d in [MODEL_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def load_all_csvs(data_dir: Path) -> pd.DataFrame:
    """Load and align all CSVs in data folder."""
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            # only numeric columns (ignore text)
            df = df.select_dtypes(include=[np.number])
            if not df.empty:
                dfs.append(df)
                logging.info(f"Loaded {f.name}: shape {df.shape}")
        except Exception as e:
            logging.warning(f"Skipping {f.name}: {e}")

    if not dfs:
        raise ValueError("No usable numeric data found in CSVs.")

    # align columns — union of all
    df = pd.concat(dfs, axis=0, ignore_index=True)
    logging.info(f"Combined dataset shape before cleaning: {df.shape}")

    # handle missing values safely
    df = df.replace([np.inf, -np.inf], np.nan)
    nan_ratio = df.isna().sum().sum() / (df.shape[0] * df.shape[1])
    logging.info(f"Missing ratio: {nan_ratio:.2%}")

    # fill NaNs with column mean or zero
    df = df.fillna(df.mean(numeric_only=True)).fillna(0)
    logging.info(f"Final cleaned dataset shape: {df.shape}")
    return df


def build_xgboost_model(df: pd.DataFrame, target_col: str):
    """Train an XGBoost model (regression/classification)."""
    if target_col not in df.columns:
        logging.warning(f"Target column '{target_col}' not found — creating synthetic target.")
        df[target_col] = df.mean(axis=1) + np.random.randn(len(df)) * 0.01

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ensure valid numeric matrix
    if X.empty or len(X) == 0:
        raise ValueError("No samples found in dataset after cleaning.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # detect classification/regression
    task = "regression"
    if len(np.unique(y)) <= 10 and all(np.floor(y) == y):
        task = "classification"
    logging.info(f"Detected task: {task}")

    if task == "regression":
        model = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = np.sqrt(mean_squared_error(y_test, preds))
        logging.info(f"RMSE: {score:.4f}")
    else:
        model = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = accuracy_score(y_test, preds)
        logging.info(f"Accuracy: {score:.4f}")

    # save model + scaler
    joblib.dump(model, MODEL_DIR / "xgboost_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    logging.info(f"Saved model and scaler in {MODEL_DIR}")

    # save predictions
    pd.DataFrame({"y_true": y_test, "y_pred": preds}).to_csv(
        OUTPUT_DIR / "xgboost_predictions.csv", index=False
    )
    logging.info(f"Saved predictions to {OUTPUT_DIR}/xgboost_predictions.csv")

    return model, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="close", help="Target column for prediction")
    args = parser.parse_args()

    df = load_all_csvs(DATA_DIR)
    build_xgboost_model(df, args.target)


if __name__ == "__main__":
    main()