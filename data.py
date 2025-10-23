# [Content of data.py remains the same]
# ... The data.py content will be included here ...
"""
Veda — Improved Data ingestion pipeline (single-file script)

This upgraded script improves robustness, provenance, and data quality for free public datasets.
Features:
 - Better handling for EEA (EU ETS) and EDGAR downloads with multiple fallbacks
 - Robust NASA POWER CSV parsing (handles commented headers)
 - NOAA CDO automatic chunking (<= 1 year per call)
 - Global Forest Watch best-effort download
 - Financial data via yfinance (graceful if not installed)
 - Manifest.csv with provenance (timestamp, source_url, filename, rows, cols, status, note)
 - Filename sanitization and safe writes
 - Summary printouts and basic validation (rows/cols/date ranges when present)

Usage:
    python veda_data_pipeline.py --start 2015-01-01 --end 2025-10-16

Optional environment variables:
 - NOAA_CDO_TOKEN (for NOAA CDO)

Notes:
 - All data fetched is real public data where available. Some official datasets require manual download or signup for free API keys; the script will provide instructions and graceful fallbacks.
"""

import os
import re
import sys
import csv
import json
import time
import math
import shutil
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from io import StringIO

import requests
import pandas as pd
from tqdm import tqdm

# optional
try:
    import yfinance as yf
except Exception:
    yf = None

# ---- CONFIG ----
DATA_DIR = Path("data").resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = DATA_DIR / "manifest.csv"
LOG_FORMAT = "%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

# safer filename
_filename_re = re.compile(r"[^0-9A-Za-z._-]")

def safe_filename(s: str) -> str:
    s = s.strip().replace(' ', '_')
    s = _filename_re.sub('_', s)
    return s


def write_manifest_entry(row: dict):
    header = [
        'retrieved_at_utc', 'dataset', 'source_url', 'filename', 'rows', 'cols', 'status', 'note'
    ]
    exists = MANIFEST.exists()
    with open(MANIFEST, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_df(df: pd.DataFrame, filename: str, source_url: str = '', dataset_name: str = ''):
    filename = safe_filename(filename)
    path = DATA_DIR / filename
    df.to_csv(path, index=True)
    logging.info(f"Saved: {path}")
    # manifest
    write_manifest_entry({
        'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': dataset_name or filename,
        'source_url': source_url,
        'filename': str(path.name),
        'rows': int(len(df)),
        'cols': int(len(df.columns)),
        'status': 'success',
        'note': ''
    })


def save_bytes(content: bytes, filename: str, source_url: str = '', dataset_name: str = ''):
    filename = safe_filename(filename)
    path = DATA_DIR / filename
    with open(path, 'wb') as f:
        f.write(content)
    logging.info(f"Saved bytes: {path}")
    # try to guess rows/cols for CSV
    rows = cols = ''
    try:
        df = pd.read_csv(path)
        rows = int(len(df)); cols = int(len(df.columns))
    except Exception:
        pass
    write_manifest_entry({
        'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': dataset_name or filename,
        'source_url': source_url,
        'filename': str(path.name),
        'rows': rows,
        'cols': cols,
        'status': 'success' if rows != '' else 'saved_bytes',
        'note': ''
    })


# ---- UTILITIES ----

def short_preview(df: pd.DataFrame, n: int = 3):
    if df is None or df.empty:
        return "(empty)"
    info = f"rows={len(df)}, cols={len(df.columns)}; columns={list(df.columns)[:10]}"
    # attempt to get date range if index is datetime-like
    try:
        if pd.api.types.is_datetime64_any_dtype(df.index):
            dr = (df.index.min(), df.index.max())
            info += f"; index_range={dr[0]} to {dr[1]}"
    except Exception:
        pass
    return info


# ---- 1) EU ETS (EEA) ----

def fetch_eu_ets_europe_data(output_basename: str = "eu_ets_eea.csv"):
    """Try multiple fallbacks to download EU ETS related CSVs. If automatic retrieval fails, the user will be informed.

    We attempt:
      1) A known EEA data-download page and parse for CSV links
      2) Several well-known mirrors / static endpoints
    """
    logging.info("Fetching EU ETS metadata page from EEA (attempting to find CSV links).")
    candidates = [
        # EEA dataset landing page (HTML). We'll parse for CSVs; not guaranteed.
        'https://www.eea.europa.eu/data-and-maps/data/european-union-emissions-trading-scheme-17/eu-ets-data-download-latest-version',
        # Common static endpoints (may change) — keep as fallbacks
        'https://www.eea.europa.eu/data-and-maps/daviz/eu-ets-auctions-html',
    ]
    found = []
    for url in candidates:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                continue
            for part in r.text.split('href="'):
                if '.csv' in part.lower():
                    link = part.split('"')[0]
                    if link.startswith('//'):
                        link = 'https:' + link
                    if link.startswith('http'):
                        found.append(link)
        except Exception as e:
            logging.debug("EEA candidate failed", exc_info=e)
    # also try a few hard-coded likely filenames (best-effort)
    hard_fallbacks = [
        'https://www.eea.europa.eu/data-and-maps/daviz/eu-ets-auctions-1/download.csv',
        # Some national/regional portals mirror EEA data — try a common pattern (may 404)
    ]
    found = list(dict.fromkeys(found + hard_fallbacks))
    if not found:
        msg = "No CSV links found automatically for EU ETS. Please visit the EEA EU ETS data page to download CSVs manually."
        logging.warning(msg)
        write_manifest_entry({
            'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
            'dataset': 'eu_ets_eea',
            'source_url': candidates[0],
            'filename': '',
            'rows': '',
            'cols': '',
            'status': 'failed',
            'note': msg
        })
        return

    # download first few links
    for i, link in enumerate(found[:4]):
        try:
            rr = requests.get(link, timeout=60)
            rr.raise_for_status()
            fname = output_basename if i == 0 else f"eu_ets_eea_part{i+1}.csv"
            save_bytes(rr.content, fname, source_url=link, dataset_name='eu_ets_eea')
        except Exception as e:
            logging.warning(f"Failed to download {link}: {e}")
            write_manifest_entry({
                'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
                'dataset': 'eu_ets_eea',
                'source_url': link,
                'filename': '',
                'rows': '',
                'cols': '',
                'status': 'failed',
                'note': str(e)
            })


# ---- 2) NASA POWER ----

def fetch_nasa_power_point(lat: float, lon: float, start: str, end: str, output_filename: str = None):
    # normalize
    start_d = pd.to_datetime(start).strftime('%Y%m%d')
    end_d = pd.to_datetime(end).strftime('%Y%m%d')
    if output_filename is None:
        output_filename = f"nasa_power_{lat}_{lon}_{start_d}_{end_d}.csv"
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?start={start_d}&end={end_d}&latitude={lat}&longitude={lon}"
        "&community=RE&parameters=ALLSKY_SFC_SW_DWN,T2M,PRECTOT"
        "&format=CSV"
    )
    logging.info("Requesting NASA POWER API (point): %s", url[:200])
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    text = r.text
    # NASA returns a CSV with comment (#) lines and variable header lines. Use pandas with comment and on_bad_lines='skip'.
    try:
        df = pd.read_csv(StringIO(text), comment='#', on_bad_lines='skip', engine='python')
    except Exception as e:
        # fallback: try to locate the first non-comment CSV header line
        lines = text.splitlines()
        header_idx = None
        for i, line in enumerate(lines[:50]):
            # a simple heuristic: header likely contains 'YYYY' or 'YEAR' or 'DATE' or 'T2M'
            if re.search(r"DATE|YEAR|T2M|ALLSKY", line, re.I):
                header_idx = i
                break
        if header_idx is None:
            raise
        csv_text = '\n'.join(lines[header_idx:])
        df = pd.read_csv(StringIO(csv_text), on_bad_lines='skip', engine='python')
    # ensure index is datetime if there is a DATE column
    for col in df.columns:
        if re.match(r"^(Date|DATE|date)$", col):
            try:
                df.index = pd.to_datetime(df[col])
                break
            except Exception:
                pass
    save_df(df, output_filename, source_url=url, dataset_name='nasa_power_point')
    return df


# ---- 3) EDGAR (GHG) ----

def fetch_edgar_ghg(output_filename: str = "edgar_ghg.csv"):
    # EDGAR often hosts datasets at JRC open-data. Try a few candidate endpoints.
    candidates = [
        'https://edgar.jrc.ec.europa.eu/dataset_ghg2024',
        'https://edgar.jrc.ec.europa.eu/overview.php?v=GHG',
    ]
    logging.info("Attempting to find an EDGAR GHG CSV (best-effort)")
    found = []
    for url in candidates:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                continue
            for part in r.text.split('href="'):
                if '.csv' in part.lower():
                    link = part.split('"')[0]
                    if link.startswith('//'):
                        link = 'https:' + link
                    if link.startswith('http'):
                        found.append(link)
        except Exception:
            continue
    if not found:
        msg = 'No direct EDGAR CSV found automatically; please download EDGAR datasets from the JRC EDGAR portal manually.'
        logging.warning(msg)
        write_manifest_entry({
            'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
            'dataset': 'edgar_ghg',
            'source_url': candidates[0],
            'filename': '',
            'rows': '',
            'cols': '',
            'status': 'failed',
            'note': msg
        })
        return
    for i, link in enumerate(found[:2]):
        try:
            rr = requests.get(link, timeout=60)
            rr.raise_for_status()
            fname = output_filename if i == 0 else f"edgar_ghg_part{i+1}.csv"
            save_bytes(rr.content, fname, source_url=link, dataset_name='edgar_ghg')
        except Exception as e:
            logging.warning(f"Failed to download EDGAR link {link}: {e}")
            write_manifest_entry({
                'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
                'dataset': 'edgar_ghg',
                'source_url': link,
                'filename': '',
                'rows': '',
                'cols': '',
                'status': 'failed',
                'note': str(e)
            })


# ---- 4) Global Forest Watch ----

def fetch_gfw_alerts(output_filename: str = "gfw_alerts.csv"):
    # Attempt a convenience CSV endpoint — open-data portal commonly exposes CSVs by dataset slug
    dataset_url = "https://data.globalforestwatch.org/datasets/gfw::integrated-deforestation-alerts.csv"
    try:
        r = requests.get(dataset_url, timeout=60)
        r.raise_for_status()
        save_bytes(r.content, output_filename, source_url=dataset_url, dataset_name='gfw_alerts')
    except Exception as e:
        msg = f"GFW CSV download failed (convenience endpoint). Please visit the GFW Open Data Portal to download datasets manually: {e}"
        logging.warning(msg)
        write_manifest_entry({
            'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
            'dataset': 'gfw_alerts',
            'source_url': dataset_url,
            'filename': '',
            'rows': '',
            'cols': '',
            'status': 'failed',
            'note': str(e)
        })


# ---- 5) Financials (yfinance) ----

def fetch_financial_symbols(symbols: list, start: str, end: str, output_prefix: str = "fin_"):
    if yf is None:
        logging.warning("yfinance is not installed. Install with: pip install yfinance")
        write_manifest_entry({
            'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
            'dataset': 'finance_yfinance',
            'source_url': '',
            'filename': '',
            'rows': '',
            'cols': '',
            'status': 'skipped',
            'note': 'yfinance not installed'
        })
        return
    for sym in symbols:
        try:
            logging.info(f"Fetching {sym} from Yahoo Finance")
            t = yf.Ticker(sym)
            df = t.history(start=start, end=end, auto_adjust=False)
            if df is None or df.empty:
                logging.warning(f"No data for {sym}")
                write_manifest_entry({
                    'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
                    'dataset': 'finance',
                    'source_url': 'yahoo_finance',
                    'filename': '',
                    'rows': 0,
                    'cols': 0,
                    'status': 'empty',
                    'note': f'No data for {sym}'
                })
                continue
            fn = f"{output_prefix}{sym}.csv"
            save_df(df, fn, source_url='yahoo_finance', dataset_name=f'finance_{sym}')
        except Exception as e:
            logging.warning(f"Failed to download {sym}: {e}")
            write_manifest_entry({
                'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
                'dataset': 'finance',
                'source_url': 'yahoo_finance',
                'filename': '',
                'rows': '',
                'cols': '',
                'status': 'failed',
                'note': str(e)
            })


# ---- 6) NOAA CDO (chunked) ----

def fetch_noaa_cdo(stationid: str, start: str, end: str, output_filename: str = None):
    token = os.getenv('NOAA_CDO_TOKEN')
    if not token:
        logging.info('NOAA token not provided. Skipping NOAA CDO fetch.')
        write_manifest_entry({
            'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
            'dataset': 'noaa_cdo',
            'source_url': 'https://www.ncdc.noaa.gov/cdo-web/',
            'filename': '',
            'rows': '',
            'cols': '',
            'status': 'skipped',
            'note': 'NOAA_CDO_TOKEN not set'
        })
        return
    if output_filename is None:
        output_filename = f"noaa_{stationid}_{start}_{end}.csv"
    base = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
    headers = {'token': token}

    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    # NOAA requires ranges <= 1 year — chunk by 1-year intervals
    rows = []
    cur_start = start_dt
    while cur_start <= end_dt:
        cur_end = min(cur_start + pd.DateOffset(years=1) - pd.Timedelta(days=1), end_dt)
        params = {
            'datasetid': 'GHCND',
            'stationid': stationid,
            'startdate': cur_start.strftime('%Y-%m-%d'),
            'enddate': cur_end.strftime('%Y-%m-%d'),
            'limit': 1000
        }
        offset = 1
        while True:
            params['offset'] = offset
            r = requests.get(base, params=params, headers=headers, timeout=30)
            if r.status_code != 200:
                logging.warning('NOAA API error %s %s', r.status_code, r.text)
                break
            data = r.json()
            results = data.get('results', [])
            if not results:
                break
            rows.extend(results)
            offset += len(results)
            if len(results) < 1000:
                break
        cur_start = cur_end + pd.Timedelta(days=1)
        time.sleep(0.2)
    if rows:
        df = pd.DataFrame(rows)
        save_df(df, output_filename, source_url=base, dataset_name='noaa_cdo')
    else:
        write_manifest_entry({
            'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
            'dataset': 'noaa_cdo',
            'source_url': base,
            'filename': '',
            'rows': 0,
            'cols': 0,
            'status': 'empty',
            'note': 'no rows returned'
        })


# ---- Orchestration / CLI ----

def main(args):
    logging.info('Starting Veda data pipeline')

    # 1) EU ETS
    fetch_eu_ets_europe_data()

    # 2) EDGAR
    fetch_edgar_ghg()

    # 3) GFW
    fetch_gfw_alerts()

    # 4) NASA POWER example point (Bengaluru)
    try:
        df_nasa = fetch_nasa_power_point(lat=12.9716, lon=77.5946, start=args.start, end=args.end, output_filename='nasa_power_bengaluru.csv')
        logging.info('NASA POWER preview: %s', short_preview(df_nasa))
    except Exception as e:
        logging.warning('NASA POWER fetch failed: %s', e)
        write_manifest_entry({
            'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
            'dataset': 'nasa_power',
            'source_url': 'https://power.larc.nasa.gov',
            'filename': '',
            'rows': '',
            'cols': '',
            'status': 'failed',
            'note': str(e)
        })

    # 5) Financials (example sustainability-linked instruments / proxies)
    symbols = [
        'ICLN',  # iShares Global Clean Energy ETF
        'TAN',   # Solar
        'KRBN',  # Carbon allowance futures ETF proxy
        '^GSPC',
        'CL=F',
        'EURUSD=X'
    ]
    fetch_financial_symbols(symbols, start=args.start, end=args.end)

    # 6) NOAA (optional)
    fetch_noaa_cdo(stationid='GHCND:INM00040971', start=args.start, end=args.end)

    logging.info('Done. Data files are in the "%s" directory. Manifest: %s', DATA_DIR, MANIFEST)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2015-01-01')
    parser.add_argument('--end', default=datetime.now(timezone.utc).strftime('%Y-%m-%d'))
    args = parser.parse_args()
    main(args)