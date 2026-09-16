"""Out-of-sample validation for the same strategy and all configured symbols."""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict

import pandas as pd

from backtest_strategy import DEFAULT_SYMBOLS, evaluate, load_data, selected_symbols
from technical_engine import StrategyConfig


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def main() -> None:
    symbols = selected_symbols() or DEFAULT_SYMBOLS
    oos_days = _env_int("OOS_DAYS", 180)
    cfg = StrategyConfig()
    results = []
    print(f"OOS start: warmup={os.environ.get('BACKTEST_DAYS', '730')} days, OOS={oos_days} days, symbols={','.join(symbols)}", flush=True)
    started = time.monotonic()

    for n, symbol in enumerate(symbols, 1):
        h1 = load_data(symbol)
        if h1.empty:
            raise RuntimeError(f"{symbol}: no H1 data")
        end = h1.index.max()
        start = end - pd.Timedelta(days=oos_days)
        print(f"[{n}/{len(symbols)}] {symbol}: OOS {start} -> {end}; bars={len(h1):,}", flush=True)
        result = evaluate(symbol, h1, cfg=cfg, trade_start=start, trade_end=end + pd.Timedelta(hours=1))
        row = asdict(result)
        row.update({
            "evaluation": "out_of_sample",
            "oos_days": oos_days,
            "oos_start": start.isoformat(),
            "oos_end": end.isoformat(),
            "warmup_days": _env_int("BACKTEST_DAYS", 730),
        })
        results.append(row)
        print(f"[{n}/{len(symbols)}] {symbol}: trades={result.trades}, win={result.win_rate_pct}%, totalR={result.total_r}, PF={result.profit_factor}", flush=True)

    with open("oos_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"OOS complete: {len(results)} symbols in {time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
