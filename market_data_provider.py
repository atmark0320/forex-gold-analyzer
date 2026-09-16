from __future__ import annotations

import os
import subprocess
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

DUKASCOPY_VERSION = "1.50.0"
SYMBOLS = {
    "USDJPY": "usdjpy", "XAUUSD": "xauusd", "EURUSD": "eurusd",
    "GBPUSD": "gbpusd", "AUDUSD": "audusd", "USDCAD": "usdcad", "USDCHF": "usdchf",
}
YFINANCE_SYMBOLS = {
    "USDJPY": "JPY=X", "XAUUSD": "XAUUSD=X", "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X", "AUDUSD": "AUDUSD=X", "USDCAD": "CAD=X", "USDCHF": "CHF=X",
}
DOWNLOAD_CHUNK_DAYS = 120
REFRESH_DAYS = int(os.environ.get("FOREX_DATA_REFRESH_DAYS", "14"))
MAX_DOWNLOAD_ATTEMPTS = 5
CACHE_DIR = Path(os.environ.get("FOREX_DATA_CACHE_DIR", ".cache/market_data"))
DUKASCOPY_CACHE_DIR = CACHE_DIR / "dukascopy_artifacts"
BATCH_SIZE = int(os.environ.get("DUKASCOPY_BATCH_SIZE", "2"))
BATCH_PAUSE_MS = int(os.environ.get("DUKASCOPY_BATCH_PAUSE_MS", "5000"))
RETRY_PAUSE_MS = int(os.environ.get("DUKASCOPY_RETRY_PAUSE_MS", "10000"))


def _run_dukascopy(instrument: str, start: datetime, end: datetime) -> pd.DataFrame:
    start = start.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    end = end.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    if end <= start:
        raise ValueError("end must be after start")
    last_error = ""
    DUKASCOPY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        try:
            with tempfile.TemporaryDirectory(prefix="dukascopy_") as tmp:
                out_dir = Path(tmp)
                file_name = f"{instrument}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
                cmd = [
                    "npx", "--yes", f"dukascopy-node@{DUKASCOPY_VERSION}", "-i", instrument,
                    "-from", start.strftime("%Y-%m-%d"), "-to", end.strftime("%Y-%m-%d"),
                    "-t", "h1", "-p", "bid", "-f", "csv", "-dir", str(out_dir), "-fn", file_name,
                    "-bs", str(BATCH_SIZE), "-bp", str(BATCH_PAUSE_MS), "-ch", "-chpath", str(DUKASCOPY_CACHE_DIR),
                    "-r", "2", "-rp", str(RETRY_PAUSE_MS), "-re", "-s",
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
                if result.returncode != 0:
                    last_error = result.stderr[-4000:] or result.stdout[-4000:]
                    if "429" in last_error and attempt < MAX_DOWNLOAD_ATTEMPTS:
                        time.sleep(RETRY_PAUSE_MS / 1000 * attempt)
                        continue
                    raise RuntimeError("Dukascopy download failed: " + last_error)
                path = out_dir / file_name
                if not path.exists() or path.stat().st_size == 0:
                    candidates = list(out_dir.glob("*.csv"))
                    if not candidates:
                        raise RuntimeError("Dukascopy returned no CSV data")
                    path = candidates[0]
                df = pd.read_csv(path)
            required = {"timestamp", "open", "high", "low", "close"}
            missing = required.difference(df.columns)
            if missing:
                raise RuntimeError(f"Dukascopy CSV columns are missing: {sorted(missing)}")
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp").sort_index()
            df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
            keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
            df = df[keep].loc[~df.index.duplicated(keep="last")]
            return df.loc[(df.index >= start) & (df.index <= end)]
        except subprocess.TimeoutExpired as exc:
            last_error = f"Dukascopy download timed out after 900s: {exc}"
            if attempt == MAX_DOWNLOAD_ATTEMPTS:
                raise RuntimeError(last_error) from exc
            time.sleep(RETRY_PAUSE_MS / 1000 * attempt)
    raise RuntimeError("Dukascopy download failed: " + last_error)


def _run_yfinance(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    ticker = YFINANCE_SYMBOLS[symbol]
    df = yf.download(
        ticker, start=start.astimezone(timezone.utc), end=end.astimezone(timezone.utc),
        interval="1h", auto_adjust=False, progress=False, threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo Finance returned no hourly data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Yahoo Finance columns are missing: {missing}")
    df.index = pd.to_datetime(df.index, utc=True)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].sort_index()
    df = df.loc[~df.index.duplicated(keep="last")]
    start_ts = pd.Timestamp(start).tz_convert("UTC")
    end_ts = pd.Timestamp(end).tz_convert("UTC")
    return df.loc[(df.index >= start_ts) & (df.index < end_ts)]


def _download_with_fallback(symbol: str, instrument: str, start: datetime, end: datetime) -> pd.DataFrame:
    try:
        return _run_dukascopy(instrument, start, end)
    except Exception as dukascopy_error:
        print(f"WARNING {symbol}: Dukascopy failed; using Yahoo Finance fallback: {dukascopy_error}", flush=True)
        try:
            return _run_yfinance(symbol, start, end)
        except Exception as fallback_error:
            raise RuntimeError(f"{symbol}: both market-data sources failed. Dukascopy={dukascopy_error}; Yahoo={fallback_error}") from fallback_error


def _drop_forming_h1(df: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    current = now or datetime.now(timezone.utc)
    current_hour = pd.Timestamp(current).tz_convert("UTC").floor("h")
    return df.loc[df.index < current_hour].copy()


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key.lower()}.csv.gz"


def _load_cached(key: str) -> pd.DataFrame | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True, compression="gzip")
        df.index = pd.to_datetime(df.index, utc=True)
        return df.sort_index()
    except Exception:
        return None


def _save_cached(key: str, df: pd.DataFrame) -> None:
    _cache_path(key).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_cache_path(key), compression="gzip", index_label="timestamp")


def fetch_ohlc(symbol: str, days: int = 450, end: datetime | None = None) -> pd.DataFrame:
    key = symbol.upper().replace("/", "")
    instrument = SYMBOLS.get(key)
    if instrument is None:
        raise ValueError(f"Unsupported market-data symbol: {symbol}")
    end = end or datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    start_ts = pd.Timestamp(start).tz_convert("UTC")
    end_ts = pd.Timestamp(end).tz_convert("UTC")
    cached = _load_cached(key)

    if cached is not None and not cached.empty and cached.index.min() <= start_ts:
        if REFRESH_DAYS > 0:
            refresh_start = max(start, end - timedelta(days=REFRESH_DAYS))
            refresh_start_ts = pd.Timestamp(refresh_start).tz_convert("UTC")
            fresh = _download_with_fallback(key, instrument, refresh_start, end)
            df = pd.concat([cached.loc[cached.index < refresh_start_ts], fresh])
        else:
            df = cached
    else:
        chunks: list[pd.DataFrame] = []
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + timedelta(days=DOWNLOAD_CHUNK_DAYS), end)
            chunk = _download_with_fallback(key, instrument, cursor, chunk_end)
            if not chunk.empty:
                chunks.append(chunk)
                combined = pd.concat(chunks).sort_index().loc[lambda x: ~x.index.duplicated(keep="last")]
                _save_cached(key, combined)
            cursor = chunk_end
            if cursor < end:
                time.sleep(5)
        df = pd.concat(chunks) if chunks else pd.DataFrame()

    df = df.sort_index().loc[~df.index.duplicated(keep="last")]
    df = _drop_forming_h1(df, end)
    df = df.loc[(df.index >= start_ts) & (df.index < end_ts)]
    if len(df) < 300:
        # A stale/incomplete cache must never be accepted as a successful backtest input.
        # Force a clean range download once, then validate the result again.
        print(f"WARNING {key}: cache produced only {len(df)} hourly bars; forcing clean refresh", flush=True)
        fresh = _download_with_fallback(key, instrument, start, end)
        fresh = _drop_forming_h1(fresh.sort_index(), end)
        fresh = fresh.loc[(fresh.index >= start_ts) & (fresh.index < end_ts)]
        if len(fresh) >= 300:
            df = fresh
            _save_cached(key, df)
    if len(df) < 300:
        raise RuntimeError(f"{symbol}: insufficient hourly data ({len(df)} bars)")
    _save_cached(key, df)
    return df


def build_timeframes(h1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = h1.copy().sort_index().loc[lambda z: ~z.index.duplicated(keep="last")]
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in x.columns:
        agg["Volume"] = "sum"
    grouped = x.resample("4h", origin="epoch", label="left", closed="left")
    counts = grouped["Close"].count()
    h4 = grouped.agg(agg).loc[counts == 4].dropna(subset=["Open", "High", "Low", "Close"])
    daily_grouped = x.resample("1D", label="left", closed="left")
    daily_counts = daily_grouped["Close"].count()
    daily = daily_grouped.agg(agg).loc[daily_counts >= 18].dropna(subset=["Open", "High", "Low", "Close"])
    return daily, h4


def latest_completed_4h_time(h1: pd.DataFrame) -> pd.Timestamp | None:
    _, h4 = build_timeframes(h1)
    return h4.index[-1] if not h4.empty else None


__all__ = ["fetch_ohlc", "build_timeframes", "latest_completed_4h_time"]
