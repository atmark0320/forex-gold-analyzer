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
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep]
    return df.loc[(df.index >= start) & (df.index <= end)]


def fetch_ohlc(symbol: str, days: int = 450, end: datetime | None = None) -> pd.DataFrame:
    """Fetch completed hourly BID candles from Dukascopy for USDJPY or XAUUSD spot."""
    key = symbol.upper().replace("/", "")
    instrument = SYMBOLS.get(key)
    if instrument is None:
        raise ValueError(f"Unsupported market-data symbol: {symbol}")
    # Never include the currently forming hourly candle.
    end = (end or datetime.now(timezone.utc)) - timedelta(hours=1)
    end = end.replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    df = _run_dukascopy(instrument, start, end)
    if len(df) < 300:
        raise RuntimeError(f"{symbol}: insufficient Dukascopy hourly data ({len(df)} bars)")
    return df


def build_timeframes(h1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in h1.columns:
        agg["Volume"] = "sum"
    h4 = h1.resample("4h").agg(agg).dropna(subset=["Open", "High", "Low", "Close"])
    daily = h1.resample("1D").agg(agg).dropna(subset=["Open", "High", "Low", "Close"])
    return daily, h4


__all__ = ["fetch_ohlc", "build_timeframes"]
