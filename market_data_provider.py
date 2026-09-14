from __future__ import annotations

import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

DUKASCOPY_VERSION = "1.50.0"
SYMBOLS = {
    "USDJPY": "usdjpy",
    "XAUUSD": "xauusd",
}


def _run_dukascopy(instrument: str, start: datetime, end: datetime) -> pd.DataFrame:
    start = start.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    end = end.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    if end <= start:
        raise ValueError("end must be after start")

    with tempfile.TemporaryDirectory(prefix="dukascopy_") as tmp:
        out_dir = Path(tmp)
        file_name = f"{instrument}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
        cmd = [
            "npx",
            "--yes",
            f"dukascopy-node@{DUKASCOPY_VERSION}",
            "-i",
            instrument,
            "-from",
            start.strftime("%Y-%m-%d"),
            "-to",
            end.strftime("%Y-%m-%d"),
            "-t",
            "h1",
            "-p",
            "bid",
            "-f",
            "csv",
            "-dir",
            str(out_dir),
            "-fn",
            file_name,
            "-ch",
            "-chpath",
            str(out_dir / ".dukascopy-cache"),
            "-r",
            "3",
            "-re",
            "-s",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if result.returncode != 0:
            raise RuntimeError(
                "Dukascopy download failed: " + (result.stderr[-2000:] or result.stdout[-2000:])
            )
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
    df = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep]
    df = df.loc[~df.index.duplicated(keep="last")]
    return df.loc[(df.index >= start) & (df.index <= end)]


def _drop_forming_h1(df: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Remove the currently-forming hourly candle so live and backtest logic match."""
    current = now or datetime.now(timezone.utc)
    current_hour = pd.Timestamp(current).tz_convert("UTC").floor("h")
    return df.loc[df.index < current_hour].copy()


def fetch_ohlc(symbol: str, days: int = 450, end: datetime | None = None) -> pd.DataFrame:
    """Fetch completed hourly BID candles from Dukascopy for USDJPY or XAUUSD spot."""
    key = symbol.upper().replace("/", "")
    instrument = SYMBOLS.get(key)
    if instrument is None:
        raise ValueError(f"Unsupported market-data symbol: {symbol}")
    end = end or datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    df = _run_dukascopy(instrument, start, end)
    df = _drop_forming_h1(df, end)
    if len(df) < 300:
        raise RuntimeError(f"{symbol}: insufficient Dukascopy hourly data ({len(df)} bars)")
    return df


def build_timeframes(h1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build only fully-formed UTC 4H candles and complete UTC daily candles."""
    x = h1.copy().sort_index()
    x = x.loc[~x.index.duplicated(keep="last")]

    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in x.columns:
        agg["Volume"] = "sum"

    # Dukascopy standard 4H candles are UTC/GMT anchored; 4H and longer periods
    # are built from hourly data. Require four completed hourly observations per 4H bar.
    grouped = x.resample("4h", origin="epoch", label="left", closed="left")
    counts = grouped["Close"].count()
    h4 = grouped.agg(agg)
    h4 = h4.loc[counts == 4].dropna(subset=["Open", "High", "Low", "Close"])

    daily_grouped = x.resample("1D", origin="epoch", label="left", closed="left")
    daily_counts = daily_grouped["Close"].count()
    daily = daily_grouped.agg(agg)
    # A normal FX trading day normally contributes many hourly observations.
    # Use at least 18 to avoid treating a weekend/holiday fragment as a daily bar.
    daily = daily.loc[daily_counts >= 18].dropna(subset=["Open", "High", "Low", "Close"])
    return daily, h4


def latest_completed_4h_time(h1: pd.DataFrame) -> pd.Timestamp | None:
    """Return the start timestamp of the newest fully completed 4H candle."""
    _, h4 = build_timeframes(h1)
    return h4.index[-1] if not h4.empty else None


__all__ = ["fetch_ohlc", "build_timeframes", "latest_completed_4h_time"]
