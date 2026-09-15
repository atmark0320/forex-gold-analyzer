from __future__ import annotations

import os
import subprocess
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

DUKASCOPY_VERSION = "1.50.0"
SYMBOLS = {
    "USDJPY": "usdjpy",
    "XAUUSD": "xauusd",
    "EURUSD": "eurusd",
    "GBPUSD": "gbpusd",
    "AUDUSD": "audusd",
    "USDCAD": "usdcad",
    "USDCHF": "usdchf",
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
                    "npx", "--yes", f"dukascopy-node@{DUKASCOPY_VERSION}",
                    "-i", instrument,
                    "-from", start.strftime("%Y-%m-%d"),
                    "-to", end.strftime("%Y-%m-%d"),
                    "-t", "h1", "-p", "bid", "-f", "csv",
                    "-dir", str(out_dir), "-fn", file_name,
                    "-bs", str(BATCH_SIZE), "-bp", str(BATCH_PAUSE_MS),
                    "-ch", "-chpath", str(DUKASCOPY_CACHE_DIR),
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
            df = df[keep]
            df = df.loc[~df.index.duplicated(keep="last")]
            return df.loc[(df.index >= start) & (df.index <= end)]
        except subprocess.TimeoutExpired as exc:
            last_error = f"Dukascopy download timed out after 900s: {exc}"
            if attempt == MAX_DOWNLOAD_ATTEMPTS:
                raise RuntimeError(last_error) from exc
            time.sleep(RETRY_PAUSE_MS / 1000 * attempt)
    raise RuntimeError("Dukascopy download failed: " + last_error)


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
    path = _cache_path(key)
    df.to_csv(path, compression="gzip", index_label="timestamp")


def fetch_ohlc(symbol: str, days: int = 450, end: datetime | None = None) -> pd.DataFrame:
    """Fetch completed hourly BID candles, reusing cached history and refreshing only when requested."""
    key = symbol.upper().replace("/", "")
    instrument = SYMBOLS.get(key)
    if instrument is None:
        raise ValueError(f"Unsupported market-data symbol: {symbol}")
    end = end or datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    cached = _load_cached(key)

    if cached is not None and not cached.empty and cached.index.min() <= pd.Timestamp(start, tz="UTC"):
        if REFRESH_DAYS > 0:
            refresh_start = max(start, end - timedelta(days=REFRESH_DAYS))
            fresh = _run_dukascopy(instrument, refresh_start, end)
            df = pd.concat([cached.loc[cached.index < pd.Timestamp(refresh_start, tz="UTC")], fresh])
        else:
            df = cached
    else:
        chunks: list[pd.DataFrame] = []
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + timedelta(days=DOWNLOAD_CHUNK_DAYS), end)
            chunk = _run_dukascopy(instrument, cursor, chunk_end)
            chunks.append(chunk)
            combined = pd.concat(chunks).sort_index()
            combined = combined.loc[~combined.index.duplicated(keep="last")]
            _save_cached(key, combined)
            cursor = chunk_end
            if cursor < end:
                time.sleep(5)
        df = pd.concat(chunks)

    df = df.sort_index()
    df = df.loc[~df.index.duplicated(keep="last")]
    df = _drop_forming_h1(df, end)
    df = df.loc[(df.index >= pd.Timestamp(start, tz="UTC")) & (df.index < pd.Timestamp(end, tz="UTC"))]
    if len(df) < 300:
        raise RuntimeError(f"{symbol}: insufficient Dukascopy hourly data ({len(df)} bars)")
    _save_cached(key, df)
    return df


def build_timeframes(h1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = h1.copy().sort_index()
    x = x.loc[~x.index.duplicated(keep="last")]
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in x.columns:
        agg["Volume"] = "sum"
    grouped = x.resample("4h", origin="epoch", label="left", closed="left")
    counts = grouped["Close"].count()
    h4 = grouped.agg(agg)
    h4 = h4.loc[counts == 4].dropna(subset=["Open", "High", "Low", "Close"])
    daily_grouped = x.resample("1D", label="left", closed="left")
    daily_counts = daily_grouped["Close"].count()
    daily = daily_grouped.agg(agg)
    daily = daily.loc[daily_counts >= 18].dropna(subset=["Open", "High", "Low", "Close"])
    return daily, h4


def latest_completed_4h_time(h1: pd.DataFrame) -> pd.Timestamp | None:
    _, h4 = build_timeframes(h1)
    return h4.index[-1] if not h4.empty else None


__all__ = ["fetch_ohlc", "build_timeframes", "latest_completed_4h_time"]
