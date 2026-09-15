"""Deterministic Dow-first backtest with bounded execution cost.

Backtest decisions are evaluated once per completed 4H candle, matching the live
strategy's decision timeframe. Entry is still executed on subsequent H1 candles.
This avoids repeatedly rebuilding the same signal four times per 4H block.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

import pandas as pd

from market_data_provider import build_timeframes, fetch_ohlc
from technical_engine import StrategyConfig, add_indicators, generate_trade_plan

DEFAULT_SYMBOLS = ("USDJPY", "XAUUSD", "EURUSD", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF")


@dataclass
class BacktestResult:
    symbol: str
    data_source: str
    trades: int
    wins: int
    losses: int
    win_rate_pct: float
    total_r: float
    expectancy_r: float
    profit_factor: float
    avg_win_r: float
    avg_loss_r: float
    max_drawdown_r: float
    max_consecutive_losses: int


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def selected_symbols() -> tuple[str, ...]:
    raw = os.environ.get("BACKTEST_SYMBOLS", "").strip()
    if not raw:
        return DEFAULT_SYMBOLS
    return tuple(x.strip().upper() for x in raw.split(",") if x.strip()) or DEFAULT_SYMBOLS


def load_data(symbol: str) -> pd.DataFrame:
    return fetch_ohlc(symbol, days=_env_int("BACKTEST_DAYS", 730))


def _trade_result(direction: str, future: pd.DataFrame, entry: float, stop: float, target: float) -> float | None:
    if future.empty:
        return None
    if direction == "buy":
        if float(future.iloc[0]["High"]) < entry:
            return None
        for _, bar in future.iterrows():
            if float(bar["Low"]) <= stop:
                return -1.0
            if float(bar["High"]) >= target:
                return abs(target - entry) / abs(entry - stop)
    else:
        if float(future.iloc[0]["Low"]) > entry:
            return None
        for _, bar in future.iterrows():
            if float(bar["High"]) >= stop:
                return -1.0
            if float(bar["Low"]) <= target:
                return abs(entry - target) / abs(stop - entry)
    return None


def evaluate(
    symbol: str,
    h1: pd.DataFrame,
    cfg: StrategyConfig = StrategyConfig(),
    max_hold_bars: int = 48,
    entry_valid_bars: int = 3,
) -> BacktestResult:
    """Evaluate only at completed 4H decision points, then execute on H1."""
    h1 = h1.sort_index().copy()
    daily_raw, h4_raw = build_timeframes(h1)

    # Indicator calculation itself is now performed once per complete frame.
    # generate_trade_plan can consume these frames without changing the trading rules.
    h1_ind = add_indicators(h1, cfg)
    h4_ind = add_indicators(h4_raw, cfg)
    daily_ind = add_indicators(daily_raw, cfg)

    # A completed 4H candle starting at T becomes actionable at the first H1
    # candle after T+4H. We use the H1 timestamp at the 4H block boundary as the
    # decision bar, which prevents forming 4H candles from entering the signal.
    decision_positions: list[int] = []
    h4_index = set(h4_ind.index)
    for i, ts in enumerate(h1_ind.index):
        if ts in h4_index and i >= 80:
            decision_positions.append(i)

    outcomes: list[float] = []
    equity = peak = max_dd = 0.0
    total = len(decision_positions)

    for n, i in enumerate(decision_positions, 1):
        decision_time = h1_ind.index[i]
        h4_hist = h4_ind.loc[:decision_time]
        daily_hist = daily_ind.loc[:decision_time]
        if len(h4_hist) < 30 or len(daily_hist) < 60:
            continue

        # Use the already-indicated H1 history. This preserves the chronological
        # information boundary while avoiding repeated indicator calculations.
        h1_hist = h1_ind.iloc[: i + 1]
        try:
            plan = generate_trade_plan(daily_hist, h4_hist, h1_hist, cfg)
        except Exception:
            continue
        if plan.direction == "wait" or plan.entry_price is None or plan.stop_price is None or plan.target1 is None:
            continue

        entry = float(plan.entry_price)
        stop = float(plan.stop_price)
        target = float(plan.target1)
        if abs(entry - stop) <= 0:
            continue

        trigger_idx = None
        search_end = min(i + 1 + entry_valid_bars, len(h1_ind))
        for j in range(i + 1, search_end):
            bar = h1_ind.iloc[j]
            if plan.direction == "buy" and float(bar["High"]) >= entry:
                trigger_idx = j
                break
            if plan.direction == "sell" and float(bar["Low"]) <= entry:
                trigger_idx = j
                break
        if trigger_idx is None:
            continue

        future = h1_ind.iloc[trigger_idx : min(trigger_idx + max_hold_bars, len(h1_ind))]
        r = _trade_result(plan.direction, future, entry, stop, target)
        if r is None:
            continue

        outcomes.append(float(r))
        equity += float(r)
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

        # Keep trades non-overlapping exactly as before.
        next_positions = [p for p in decision_positions if p >= trigger_idx + max_hold_bars]
        if next_positions:
            decision_positions = decision_positions[:n] + next_positions
            total = len(decision_positions)

        if n % 250 == 0:
            print(f"    {symbol}: decisions {n:,}, trades {len(outcomes):,}", flush=True)

    wins = [x for x in outcomes if x > 0]
    losses = [x for x in outcomes if x < 0]
    trades = len(outcomes)
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    streak = max_streak = 0
    for x in outcomes:
        if x < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    return BacktestResult(
        symbol=symbol,
        data_source="Dukascopy spot BID 1H",
        trades=trades,
        wins=len(wins),
        losses=len(losses),
        win_rate_pct=round(100 * len(wins) / trades, 2) if trades else 0.0,
        total_r=round(sum(outcomes), 3),
        expectancy_r=round(sum(outcomes) / trades, 4) if trades else 0.0,
        profit_factor=round(gross_profit / gross_loss, 3) if gross_loss else (999.0 if gross_profit else 0.0),
        avg_win_r=round(gross_profit / len(wins), 4) if wins else 0.0,
        avg_loss_r=round(sum(losses) / len(losses), 4) if losses else 0.0,
        max_drawdown_r=round(max_dd, 3),
        max_consecutive_losses=max_streak,
    )


def main() -> None:
    symbols = selected_symbols()
    days = _env_int("BACKTEST_DAYS", 730)
    started = time.monotonic()
    results = []
    print(f"BACKTEST start: days={days}, symbols={','.join(symbols)}", flush=True)

    for n, symbol in enumerate(symbols, 1):
        symbol_started = time.monotonic()
        print(f"[{n}/{len(symbols)}] loading {symbol}...", flush=True)
        h1 = load_data(symbol)
        print(f"[{n}/{len(symbols)}] {symbol}: {len(h1):,} H1 bars; evaluating completed 4H decisions...", flush=True)
        result = evaluate(symbol, h1)
        results.append(asdict(result))
        print(
            f"[{n}/{len(symbols)}] {symbol}: trades={result.trades}, total_R={result.total_r}, "
            f"expectancy={result.expectancy_r}, elapsed={time.monotonic() - symbol_started:.1f}s",
            flush=True,
        )

    payload = json.dumps(results, ensure_ascii=False, indent=2)
    with open("backtest_results.json", "w", encoding="utf-8") as f:
        f.write(payload)
    print(payload, flush=True)
    print(f"BACKTEST complete: elapsed={time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
