#!/usr/bin/env python3
"""
predictions.py — Combine XGBoost predictions with sentiment and produce final verdict
------------------------------------------------------------------------------
Loads:
 - outputs/xgboost_predictions.csv  (must contain y_true, y_pred)
 - outputs/sentiment_summary.csv    (should contain timestamp, sentiment_score, sentiment_label)
 - models/xgboost_model.pkl         (optional — not required for ensembling)

Creates a second-level XGBoost that combines the base model's predictions and the
sentiment score (plus simple engineered features). If there is not enough data to
train a second-level model, falls back to a robust weighted ensemble.

Outputs:
 - models/combination_xgb.pkl
 - outputs/combined_predictions.csv
 - A matplotlib plot showing actual, base predicted, and combined predicted values

Usage:
    python predictions.py --future 30 --retrain

Options:
  --future N     : project N future steps using linear trend on combined preds
  --retrain      : force retrain of the combiner model if enough samples
  --plot         : show the plot (default: False on server environments)

"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Fix for headless environments: use Agg backend if plotting is enabled but no display
import matplotlib
try:
    matplotlib.use('Agg') # Use non-GUI backend for server environments
    import matplotlib.pyplot as plt
    PLOTTING_ENABLED = True
except ImportError:
    PLOTTING_ENABLED = False
    logging.warning("Matplotlib not found or display not available. Plotting will be skipped.")
    

from datetime import datetime

# --- Paths ---
BASE_DIR = Path('.')
OUTPUT_DIR = BASE_DIR / 'outputs'
MODEL_DIR = BASE_DIR / 'models'
PRED_FILE = OUTPUT_DIR / 'xgboost_predictions.csv'
SENT_FILE = OUTPUT_DIR / 'sentiment_summary.csv'
COMBO_MODEL_FILE = MODEL_DIR / 'combination_xgb.pkl'
COMBO_PRED_OUT = OUTPUT_DIR / 'combined_predictions.csv'

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def load_files():
    if not PRED_FILE.exists():
        raise FileNotFoundError(f"Prediction file not found: {PRED_FILE}. Run xgboost_model.py first.")
    df_pred = pd.read_csv(PRED_FILE)
    if 'y_true' not in df_pred.columns or 'y_pred' not in df_pred.columns:
        raise ValueError("xgboost_predictions.csv must contain 'y_true' and 'y_pred' columns")

    # load sentiment (may be summary or historical)
    if SENT_FILE.exists():
        df_sent = pd.read_csv(SENT_FILE)
    else:
        logging.warning(f"Sentiment file not found: {SENT_FILE}. Using neutral sentiment=0.5")
        df_sent = pd.DataFrame([{'timestamp': pd.Timestamp.utcnow(), 'sentiment_score': 0.5, 'sentiment_label': 'NEUTRAL'}])

    return df_pred, df_sent


def align_sentiment(df_pred: pd.DataFrame, df_sent: pd.DataFrame) -> pd.DataFrame:
    """Attach sentiment to predictions. If sentiment has many rows and matches index/length,
    attempt to align by time; otherwise broadcast latest sentiment to all rows."""
    df = df_pred.copy().reset_index(drop=True)

    # If sentiment file has many rows and length matches, just align by nearest index
    if len(df_sent) == len(df):
        if 'sentiment_score' in df_sent.columns:
            df['sentiment_score'] = df_sent['sentiment_score'].values
            df['sentiment_label'] = df_sent.get('sentiment_label', pd.Series(['NEUTRAL']*len(df)))
            logging.info('Aligned sentiment by row count (equal lengths).')
            return df

    # If sentiment has timestamp column and more rows, try temporal join
    if 'timestamp' in df_sent.columns and df_sent.shape[0] > 1:
        try:
            s = df_sent.copy()
            s['timestamp'] = pd.to_datetime(s['timestamp'])
            # assume df_pred may not have timestamps; if not, broadcast latest
        except Exception:
            pass

    # default: use latest sentiment summary (last row)
    latest = df_sent.iloc[-1]
    score = float(latest.get('sentiment_score', 0.5))
    label = latest.get('sentiment_label', 'NEUTRAL')
    df['sentiment_score'] = score
    df['sentiment_label'] = label
    logging.info(f'Broadcasting latest sentiment: {label} ({score:.3f}) to all prediction rows')
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create meta-features for the combiner model."""
    df2 = df.copy()
    # base prediction
    df2['pred'] = df2['y_pred']
    df2['true'] = df2['y_true']
    # error of base model
    df2['residual'] = df2['true'] - df2['pred']
    df2['abs_residual'] = df2['residual'].abs()
    # rolling stats — if not enough rows, will compute on available
    window = min(10, max(1, len(df2)//5))
    df2['pred_roll_mean'] = df2['pred'].rolling(window=window, min_periods=1).mean()
    df2['pred_roll_std'] = df2['pred'].rolling(window=window, min_periods=1).std().fillna(0)
    # sentiment
    df2['sentiment_score'] = df2['sentiment_score'].astype(float)
    # interaction features
    df2['pred_x_sent'] = df2['pred'] * df2['sentiment_score']
    df2['resid_x_sent'] = df2['residual'] * df2['sentiment_score']

    # final feature set
    features = [
        'pred', 'pred_roll_mean', 'pred_roll_std',
        'sentiment_score', 'pred_x_sent', 'resid_x_sent', 'abs_residual'
    ]
    # ensure features exist (fill missing)
    for c in features:
        if c not in df2.columns:
            df2[c] = 0.0
    X = df2[features].fillna(0.0)
    return X, df2


def train_combiner(X, y, retrain=False):
    """Train or load an XGBoost combiner model. Return model and a flag whether trained."""
    # if existing model and not forcing retrain, load it
    if COMBO_MODEL_FILE.exists() and not retrain:
        try:
            model = joblib.load(COMBO_MODEL_FILE)
            logging.info(f'Loaded existing combiner model from {COMBO_MODEL_FILE}')
            return model, False
        except Exception as e:
            logging.warning(f'Failed to load existing combiner model: {e} — will retrain')

    # need at least some samples to train
    if len(X) < 30:
        logging.warning('Not enough samples to train combiner (need >=30 rows). Skipping training.')
        return None, False

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # save
    joblib.dump(model, COMBO_MODEL_FILE)
    logging.info(f'Trained combiner model and saved to {COMBO_MODEL_FILE}')
    return model, True


def fallback_ensemble(df):
    """If we cannot train a combiner, use a robust weighted ensemble.
    We'll weight base prediction and scaled sentiment score.
    sentiment_score in [0,1] -> map to price scale via mean(true)
    final = w1 * pred + w2 * sentiment_mapped
    where weights chosen by simple heuristic.
    """
    df2 = df.copy()
    mean_true = df2['y_true'].abs().mean() if df2['y_true'].abs().mean() != 0 else 1.0
    # map sentiment into price-like value centered around mean of pred
    sentiment_mapped = (df2['sentiment_score'] - 0.5) * mean_true * 0.2  # small effect
    w_pred = 0.75
    w_sent = 0.25
    df2['combined_pred'] = w_pred * df2['y_pred'] + w_sent * (df2['y_pred'] + sentiment_mapped)
    return df2


def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mean_val = np.mean(np.abs(y_true)) if np.mean(np.abs(y_true)) != 0 else 1.0
    accuracy_pct = max(0.0, 100.0 * (1 - rmse / mean_val))
    residuals = np.abs(y_true - y_pred)
    conf = max(0.0, 100.0 * (1 - np.var(residuals) / (np.var(y_true) + 1e-9)))
    return rmse, accuracy_pct, conf


def sentiment_label_from_score(score):
    if score > 0.55:
        return 'Bullish'
    elif score < 0.45:
        return 'Bearish'
    else:
        return 'Neutral'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--future', type=int, default=0, help='Project N future steps')
    # CHANGED DEFAULT: Plotting is now opt-in for server readiness
    parser.add_argument('--plot', action='store_true', default=False, help='Show plot (requires display environment)')
    parser.add_argument('--retrain', action='store_true', help='Force retrain of combiner model')
    args = parser.parse_args()

    df_pred, df_sent = load_files()
    df = align_sentiment(df_pred, df_sent)
    X, df_feat = engineer_features(df)
    y = df_feat['true'].values

    # try to train combiner
    model, trained = train_combiner(X, y, retrain=args.retrain)

    if model is not None:
        df_feat['combined_pred'] = model.predict(X)
    else:
        df_feat = fallback_ensemble(df_feat)
        logging.info('Used fallback weighted ensemble for combined predictions.')

    # save combined predictions
    out_df = df_feat[['true', 'y_pred', 'combined_pred', 'sentiment_score', 'sentiment_label']].copy()
    out_df = out_df.rename(columns={'true': 'y_true'})
    out_df.to_csv(COMBO_PRED_OUT, index=False)
    logging.info(f'Saved combined predictions to {COMBO_PRED_OUT}')

    # metrics
    rmse, acc_pct, conf = compute_metrics(out_df['y_true'].values, out_df['combined_pred'].values)
    sentiment_overall = sentiment_label_from_score(df_feat['sentiment_score'].iloc[-1])
    logging.info(f'Final RMSE: {rmse:.6f} | Accuracy: {acc_pct:.2f}% | Confidence: {conf:.2f}% | Sentiment: {sentiment_overall}')

    # Plotting is now conditional on the PLOTTING_ENABLED flag and --plot argument
    if args.plot and PLOTTING_ENABLED:
        plt.figure(figsize=(12, 6))
        x = np.arange(len(out_df))
        plt.plot(x, out_df['y_true'], label='Actual')
        plt.plot(x, out_df['y_pred'], label='Base Predicted', linestyle='--')
        plt.plot(x, out_df['combined_pred'], label='Combined Predicted', linestyle='-.')
        if args.future > 0:
            # project
            last = out_df['combined_pred'].iloc[-10:]
            if len(last) >= 2:
                trend = np.polyfit(np.arange(len(last)), last, 1)
                future_x = np.arange(len(out_df), len(out_df) + args.future)
                future_y = np.polyval(trend, np.arange(len(last), len(last) + args.future))
                plt.plot(np.concatenate([x, future_x]), np.concatenate([out_df['combined_pred'], future_y]),
                         label=f'{args.future}-step Forecast', linestyle=':')
                plt.axvspan(len(out_df)-1, len(out_df)+args.future, color='lightgrey', alpha=0.25)

        text = (f"Accuracy: {acc_pct:.2f}%\n"
                f"Confidence: {conf:.2f}%\n"
                f"RMSE: {rmse:.6f}\n"
                f"Sentiment: {sentiment_overall} ({df_feat['sentiment_score'].iloc[-1]:.3f})")
        plt.gcf().text(0.02, 0.75, text, fontsize=11, bbox=dict(facecolor='white', alpha=0.9, edgecolor='grey'))
        plt.title('Final Combined Market Prediction')
        plt.xlabel('Sample / Time step')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        # Save plot to file instead of showing it (best practice for server environment)
        # We save it to a temporary file that a local user could inspect if needed.
        plt.savefig(OUTPUT_DIR / "combined_predictions_plot.png")
        logging.info(f"Saved plot to {OUTPUT_DIR}/combined_predictions_plot.png")


if __name__ == '__main__':
    main()
