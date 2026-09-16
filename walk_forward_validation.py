"""Rolling multi-window out-of-sample validation.

Uses the existing deterministic backtest and selected test-only candidate
variants. Seven 180-day windows, spaced 90 days apart, are evaluated from the
same 730-day dataset. The reduced candidate set keeps the validation within
GitHub Actions limits while preserving the baseline and the structural
filters that showed the most useful signal in the prior OOS screen.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from backtest_strategy import DEFAULT_SYMBOLS, load_data
from strategy_candidates import CANDIDATES, ADX_GATES, FULL_ALIGNMENT, RISING_ADX, evaluate_candidate

WINDOW_DAYS = 180
STEP_DAYS = 90
WINDOWS = 7

# Keep the rolling test focused on the candidates that materially changed
# trade selection in the previous OOS screen.
VARIANTS = (
    "baseline",
    "adx25_gate",
    "adx25_faster_target",
    "full_alignment_adx25",
    "adx25_rising",
    "adx25_rising_faster_target",
)


def main() -> None:
    missing = [name for name in VARIANTS if name not in CANDIDATES]
    if missing:
        raise RuntimeError(f"Unknown rolling variants: {missing}")

    all_rows = []
    for symbol in DEFAULT_SYMBOLS:
        print(f"Loading {symbol}...", flush=True)
        h1 = load_data(symbol)
        end = h1.index.max()
        for window_no in range(WINDOWS):
            window_end = end - pd.Timedelta(days=STEP_DAYS * window_no)
            window_start = window_end - pd.Timedelta(days=WINDOW_DAYS)
            print(
                f"{symbol} window={window_no + 1} "
                f"{window_start.isoformat()} -> {window_end.isoformat()}",
                flush=True,
            )
            for name in VARIANTS:
                cfg = CANDIDATES[name]
                result = evaluate_candidate(
                    symbol,
                    h1,
                    cfg=cfg,
                    adx_gate=ADX_GATES.get(name),
                    full_alignment=name in FULL_ALIGNMENT,
                    rising_adx=name in RISING_ADX,
                    trade_start=window_start,
                    trade_end=window_end + pd.Timedelta(hours=1),
                )
                row = {
                    **result.__dict__,
                    "variant": name,
                    "symbol": symbol,
                    "window": window_no + 1,
                    "window_start": window_start.isoformat(),
                    "window_end": window_end.isoformat(),
                    "evaluation": "rolling_out_of_sample",
                    "oos_days": WINDOW_DAYS,
                    "adx_gate": ADX_GATES.get(name),
                    "full_alignment": name in FULL_ALIGNMENT,
                    "rising_adx": name in RISING_ADX,
                }
                all_rows.append(row)

    Path("walk_forward_results.json").write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Stability summary: aggregate R and trade count by symbol/variant, and
    # count profitable windows. No ranking by a single window is used.
    summary = []
    frame = pd.DataFrame(all_rows)
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

    Path("walk_forward_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Walk-forward validation complete: {len(all_rows)} rows", flush=True)


if __name__ == "__main__":
    main()
