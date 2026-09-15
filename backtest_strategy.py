"""Fast, deterministic backtest for the Dow-first strategy engine.

The important design point is that indicators and Dow snapshots are computed once
per timeframe. The old implementation rebuilt every historical dataframe and
recomputed all indicators on every H1 decision bar, which made a 730-day,
7-symbol run unnecessarily expensive.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

import pandas as pd

from market_data_provider import build_timeframes, fetch_ohlc
from technical_engine import StrategyConfig, add_indicators, dow_structure, generate_trade_plan

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
    wanted = tuple(x.strip().upper() for x in raw.split(",") if x.strip())
    return wanted or DEFAULT_SYMBOLS


def load_data(symbol: str) -> pd.DataFrame:
    return fetch_ohlc(symbol, days=_env_int("BACKTEST_DAYS", 730))


def _trade_result(direction: str, future: pd.DataFrame, entry: float, stop: float, target: float) -> float | None:
    if future.empty:
        return None
    if direction == "buy":
        if float(future.iloc[0]["High"]) < entry:
            return None
        for _, bar in future.iterrows():
            # Conservative ordering when both are touched in one bar.
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


def _prepare_frames(h1: pd.DataFrame, cfg: StrategyConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare H1/H4/D1 indicators once instead of once per decision bar."""
    daily_raw, h4_raw = build_timeframes(h1)
    h1_ind = add_indicators(h1, cfg)
    h4_ind = add_indicators(h4_raw, cfg)
    daily_ind = add_indicators(daily_raw, cfg)
    return h1_ind, h4_ind, daily_ind


def _historical_slice(frame: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
    return frame.loc[:timestamp]


def evaluate(
    symbol: str,
    h1: pd.DataFrame,
    cfg: StrategyConfig = StrategyConfig(),
    max_hold_bars: int = 48,
    entry_valid_bars: int = 3,
) -> BacktestResult:
    """Chronological, non-overlapping evaluation using only information available at decision time."""
    h1, h4_full, daily_full = _prepare_frames(h1.sort_index().copy(), cfg)
    outcomes: list[float] = []
    equity = peak = max_dd = 0.0
    i = 80
    total = len(h1)

    while i < total - 2:
        decision_time = h1.index[i]
        h4_hist = _historical_slice(h4_full, decision_time)
        daily_hist = _historical_slice(daily_full, decision_time)
        if len(h4_hist) < 30 or len(daily_hist) < 60:
            i += 1
            continue

        # generate_trade_plan still owns the exact trading rules. Passing already
        # calculated frames makes the expensive indicator stage cheap.
        h1_hist = h1.iloc[: i + 1]
        try:
            plan = generate_trade_plan(daily_hist, h4_hist, h1_hist, cfg)
        except Exception:
            i += 1
            continue
        if plan.direction == "wait" or plan.entry_price is None or plan.stop_price is None or plan.target1 is None:
            i += 1
            continue

        entry = float(plan.entry_price)
        stop = float(plan.stop_price)
        target = float(plan.target1)
        if abs(entry - stop) <= 0:
            i += 1
            continue

        trigger_idx = None
        search_end = min(i + 1 + entry_valid_bars, total)
        for j in range(i + 1, search_end):
            bar = h1.iloc[j]
            if plan.direction == "buy" and float(bar["High"]) >= entry:
                trigger_idx = j
                break
            if plan.direction == "sell" and float(bar["Low"]) <= entry:
                trigger_idx = j
                break
        if trigger_idx is None:
            i += 1
            continue

        future = h1.iloc[trigger_idx : min(trigger_idx + max_hold_bars, total)]
        r = _trade_result(plan.direction, future, entry, stop, target)
        if r is None:
            i = max(i + 1, trigger_idx + max_hold_bars)
            continue

        outcomes.append(float(r))
        equity += float(r)
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
        i = max(i + 1, trigger_idx + max_hold_bars)

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
        print(f"[{n}/{len(symbols)}] {symbol}: {len(h1):,} H1 bars; evaluating...", flush=True)
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
