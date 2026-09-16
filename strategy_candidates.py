"""All-symbol strategy candidate screening on the same out-of-sample window.

This module deliberately keeps the baseline and every candidate result. It does
not select a winner automatically; the next promotion step is based on both
IS and OOS evidence across all seven symbols.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from pathlib import Path

from backtest_strategy import DEFAULT_SYMBOLS, evaluate, load_data
from technical_engine import StrategyConfig

CANDIDATES = {
    "baseline": StrategyConfig(),
    # Require one additional confirmation point before entering.
    "stricter_confirmation": replace(StrategyConfig(), min_score=8, strong_score=10),
    # Give trend continuation more room while keeping the same Dow-first entry logic.
    "wider_target": replace(StrategyConfig(), target1_r=1.8, target2_r=2.8),
    # Reduce target distance to test whether the baseline is losing too many
    # trades to reversals before reaching 1.5R.
    "faster_target": replace(StrategyConfig(), target1_r=1.25, target2_r=2.0),
    # Test a wider volatility stop without changing the signal conditions.
    "wider_stop": replace(StrategyConfig(), stop_atr=1.60),
    # Require a slightly stronger breakout displacement before execution.
    "larger_entry_buffer": replace(StrategyConfig(), entry_buffer_atr=0.15),
}


def oos_days() -> int:
    try:
        return max(30, int(os.environ.get("OOS_DAYS", "180")))
    except ValueError:
        return 180


def main() -> None:
    days = oos_days()
    symbols = DEFAULT_SYMBOLS
    output = []
    for symbol in symbols:
        print(f"Loading {symbol}...", flush=True)
        h1 = load_data(symbol)
        end = h1.index.max()
        start = end - __import__("pandas").Timedelta(days=days)
        for name, cfg in CANDIDATES.items():
            result = evaluate(
                symbol,
                h1,
                cfg=cfg,
                trade_start=start,
                trade_end=end + __import__("pandas").Timedelta(hours=1),
            )
            row = asdict(result)
            row.update({"variant": name, "evaluation": "out_of_sample", "oos_days": days})
            output.append(row)
            print(
                f"{symbol} / {name}: trades={result.trades}, "
                f"win={result.win_rate_pct}%, totalR={result.total_r}, "
                f"PF={result.profit_factor}, DD={result.max_drawdown_r}",
                flush=True,
            )

    Path("strategy_candidates_oos.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Candidate OOS complete: {len(output)} rows", flush=True)


if __name__ == "__main__":
    main()
