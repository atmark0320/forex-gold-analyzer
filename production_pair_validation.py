"""Pair-specific rolling validation for production candidates.

This is deliberately separate from the broad 7-symbol research screen. It
compares the current production baseline with the XAUUSD candidate using the
same completed-bar backtest and the same rolling windows.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from backtest_strategy import load_data, evaluate
from strategy_candidates import CANDIDATES, ADX_GATES, evaluate_candidate

WINDOW_DAYS = 180
STEP_DAYS = 90
WINDOWS = 7

CASES = (
    ("USDJPY", "baseline"),
    ("XAUUSD", "baseline"),
    ("XAUUSD", "adx25_faster_target"),
)


def main() -> None:
    rows = []
    for symbol, variant in CASES:
        print(f"Loading {symbol}...", flush=True)
        h1 = load_data(symbol)
        end = h1.index.max()
        for window in range(1, WINDOWS + 1):
            window_end = end - pd.Timedelta(days=STEP_DAYS * (window - 1))
            window_start = window_end - pd.Timedelta(days=WINDOW_DAYS)
            cfg = CANDIDATES[variant]
            result = evaluate_candidate(
                symbol,
                h1,
                cfg=cfg,
                adx_gate=ADX_GATES.get(variant),
                trade_start=window_start,
                trade_end=window_end + pd.Timedelta(hours=1),
            )
            rows.append({
                **result.__dict__,
                "symbol": symbol,
                "variant": variant,
                "window": window,
                "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(),
                "evaluation": "production_pair_rolling_oos",
            })
            print(
                f"{symbol} / {variant} / window={window}: "
                f"trades={result.trades}, totalR={result.total_r}, "
                f"PF={result.profit_factor}, DD={result.max_drawdown_r}",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    summary = []
    for (symbol, variant), group in frame.groupby(["symbol", "variant"], sort=False):
        summary.append({
            "symbol": symbol,
            "variant": variant,
            "windows": int(len(group)),
            "positive_windows": int((group["total_r"] > 0).sum()),
            "negative_windows": int((group["total_r"] < 0).sum()),
            "total_r": round(float(group["total_r"].sum()), 3),
            "median_window_r": round(float(group["total_r"].median()), 3),
            "worst_window_r": round(float(group["total_r"].min()), 3),
            "best_window_r": round(float(group["total_r"].max()), 3),
            "total_trades": int(group["trades"].sum()),
        })

    Path("production_pair_validation.json").write_text(
        json.dumps({"rows": rows, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Production pair validation complete", flush=True)


if __name__ == "__main__":
    main()
