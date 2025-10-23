# [Content of sentimental.py remains the same]
# ... The sentimental.py content will be included here ...
#!/usr/bin/env python3
"""
sentimental.py — Web Sentiment Scraper + BERT Sentiment Analyzer
----------------------------------------------------------------
Scrapes multiple financial and news websites for textual data,
runs sentiment analysis using a BERT model (finance-tuned if available),
and outputs both detailed and summarized sentiment CSVs.

FIXED: Implemented robust fetching with a longer random delay and retries
to mitigate 401/403/404 errors from aggressive web scraping defense.

Usage:
    python sentimental.py --output outputs/sentiment_summary.csv
    python sentimental.py --schedule 60   # (optional) re-run every 60 mins
"""

import os
import time
import random
import logging
import argparse
import requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from datetime import datetime
from transformers import pipeline

# --- Setup ---
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# ---- CONFIG ----
HEADERS_LIST = [
    # Modern browser agents
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/120.0",
]
# Increased delay to reduce risk of blocking
REQUEST_DELAY_SEC = (4, 8)
MAX_RETRIES = 3

# Use a finance-specific BERT sentiment model if available
MODEL_NAME = "ProsusAI/finbert"  # FinBERT trained on financial text
# Initialize pipeline only once
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model=MODEL_NAME)
except Exception as e:
    logging.warning(f"Could not load FinBERT. Falling back to default: {e}")
    sentiment_pipeline = pipeline("sentiment-analysis")


# ---- SOURCES ----
SOURCES = [
    {
        "name": "Reuters",
        "url": "https://www.reuters.com/markets/",
        "parser": lambda soup: [p.get_text().strip() for p in soup.select("p") if len(p.get_text()) > 60]
    },
    {
        "name": "Investing",
        "url": "https://www.investing.com/news/stock-market-news",
        "parser": lambda soup: [h.text.strip() for h in soup.select("article a") if len(h.text.strip()) > 30]
    },
    {
        "name": "Bloomberg",
        "url": "https://www.bloomberg.com/markets",
        # NOTE: Bloomberg is often highly protected. Using general text selection.
        "parser": lambda soup: [p.text.strip() for p in soup.select("p") if len(p.text) > 50]
    },
]


# ---- FUNCTIONS ----

def fetch_url(url: str) -> str:
    """Fetch page HTML with random user-agent, delay, and retries."""
    for attempt in range(MAX_RETRIES):
        headers = {"User-Agent": random.choice(HEADERS_LIST)}
        time.sleep(random.uniform(*REQUEST_DELAY_SEC))
        try:
            r = requests.get(url, headers=headers, timeout=20)
            r.raise_for_status()
            return r.text
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                logging.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed for {url}: {e}. Retrying...")
            else:
                raise e
    return ""


def parse_source(source: dict) -> pd.DataFrame:
    """Fetch, parse and return extracted text as DataFrame."""
    try:
        logging.info(f"Fetching: {source['name']} ({source['url']})")
        html = fetch_url(source["url"])
        soup = BeautifulSoup(html, "html.parser")
        texts = source["parser"](soup)
        df = pd.DataFrame({
            "source": source["name"],
            "datetime": datetime.utcnow(),
            "text": texts
        })
        logging.info(f"{source['name']}: extracted {len(df)} text blocks")
        return df
    except Exception as e:
        # Catch exception from fetch_url when all retries fail
        logging.warning(f"Failed to fetch {source['name']} after {MAX_RETRIES} attempts: {e}")
        return pd.DataFrame(columns=["source", "datetime", "text"])


def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Run FinBERT/BERT sentiment model on texts."""
    if df.empty:
        return df
    sentiments, scores = [], []
    for t in df["text"]:
        try:
            # BERT models have a max input length (e.g., 512 tokens)
            result = sentiment_pipeline(t[:512])[0]
            # FinBERT's output labels are POSITIVE, NEGATIVE, NEUTRAL
            label = result["label"]
            score = result["score"]
            # To normalize score (0.5 for neutral, 1 for strong positive, 0 for strong negative)
            if label == "NEUTRAL":
                # Neutral score remains 0.5 regardless of magnitude
                normalized_score = 0.5
            elif label == "POSITIVE":
                # Scale positive score between 0.5 and 1.0
                normalized_score = 0.5 + score / 2
            else: # NEGATIVE
                # Scale negative score between 0.0 and 0.5
                normalized_score = 0.5 - score / 2

            sentiments.append(label)
            scores.append(normalized_score)

        except Exception as e:
            sentiments.append("NEUTRAL")
            scores.append(0.5) # Default neutral score for failed analysis
            logging.debug(f"Failed on text: {e}")
    df["sentiment_label"] = sentiments
    df["sentiment_score"] = scores
    return df


def aggregate_sentiment(df: pd.DataFrame) -> dict:
    """Aggregate to an overall market sentiment."""
    if df.empty:
        return {"sentiment_score": 0.5, "sentiment_label": "NEUTRAL"} # Default to neutral

    # Calculate the mean of the normalized sentiment scores
    mean_score = df["sentiment_score"].mean()

    # Determine the overall label based on the mean score
    if mean_score >= 0.55: # Threshold for positive
        dominant = "POSITIVE"
    elif mean_score <= 0.45: # Threshold for negative
        dominant = "NEGATIVE"
    else:
        dominant = "NEUTRAL"

    # Use the calculated mean score and the derived dominant label
    return {"sentiment_score": mean_score, "sentiment_label": dominant}


def run_sentiment_scraper(output_path: Path):
    all_data = []
    for src in SOURCES:
        df_src = parse_source(src)
        if not df_src.empty:
            df_src = analyze_sentiment(df_src)
            all_data.append(df_src)

    if not all_data:
        logging.error("No data retrieved from any source. Sentiment defaulted to NEUTRAL.")
        result = {"sentiment_score": 0.5, "sentiment_label": "NEUTRAL"}
        df_all = pd.DataFrame()
    else:
        df_all = pd.concat(all_data, ignore_index=True)
        result = aggregate_sentiment(df_all)

    # save detailed and summary outputs
    df_all.to_csv(output_path.with_name("sentiment_details.csv"), index=False)
    summary_df = pd.DataFrame([{
        "timestamp": datetime.utcnow(),
        "sentiment_score": result["sentiment_score"],
        "sentiment_label": result["sentiment_label"],
        "num_texts": len(df_all)
    }])
    summary_df.to_csv(output_path, index=False)

    logging.info(f"✅ Sentiment Summary: {result['sentiment_label']} ({result['sentiment_score']:.2f})")
    logging.info(f"Saved to {output_path} and sentiment_details.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_DIR / "sentiment_summary.csv"),
                        help="Path to save sentiment summary CSV")
    parser.add_argument("--schedule", type=int, default=0,
                        help="If >0: schedule scraper every N minutes")
    args = parser.parse_args()

    if args.schedule > 0:
        import schedule
        logging.info(f"Scheduling scraper every {args.schedule} minutes")
        schedule.every(args.schedule).minutes.do(lambda: run_sentiment_scraper(Path(args.output)))
        while True:
            schedule.run_pending()
            time.sleep(30)
    else:
        run_sentiment_scraper(Path(args.output))


if __name__ == "__main__":
    main()