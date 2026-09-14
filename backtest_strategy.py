"""Walk-forward evaluator for the deterministic Dow-first strategy engine."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import pandas as pd
import yfinance as yf

from technical_engine import StrategyConfig, generate_trade_plan, resample_ohlc


@dataclass
class BacktestResult:
    symbol: str
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


def load_data(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = yf.Ticker(symbol).history(period="2y", interval="1d", auto_adjust=False)
    h1 = yf.Ticker(symbol).history(period="60d", interval="1h", auto_adjust=False)
    if daily.empty or h1.empty:
        raise RuntimeError(f"No data for {symbol}")
    return daily, h1


def _trade_result(plan, future: pd.DataFrame, entry: float, stop: float, target: float) -> float | None:
    if future.empty:
        return None
    if plan.direction == "buy":
        if float(future.iloc[0]["High"]) < entry:
            return None
        for _, bar in future.iterrows():
            hit_sl = float(bar["Low"]) <= stop
            hit_tp = float(bar["High"]) >= target
            if hit_sl:
                # Conservative convention: if SL and TP are both touched in the same bar,
                # assume the stop was hit first because intrabar ordering is unknown.
                return -1.0
            if hit_tp:
                return abs(target - entry) / abs(entry - stop)
    else:
        if float(future.iloc[0]["Low"]) > entry:
            return None
        for _, bar in future.iterrows():
            hit_sl = float(bar["High"]) >= stop
            hit_tp = float(bar["Low"]) <= target
            if hit_sl:
                return -1.0
            if hit_tp:
                return abs(entry - target) / abs(stop - entry)
    return None


def evaluate(
    symbol: str,
    daily: pd.DataFrame,
    h1: pd.DataFrame,
    cfg: StrategyConfig = StrategyConfig(),
    max_hold_bars: int = 48,
    entry_valid_bars: int = 3,
) -> BacktestResult:
    """Chronological, non-overlapping evaluation.

    At each decision point only bars available at that time are passed to the strategy.
    A signal can trigger within ``entry_valid_bars`` future candles. Once entered, the
    evaluator waits until the trade resolves before looking for a new one.
    """
    h1 = h1.sort_index().copy()
    h4 = resample_ohlc(h1, "4h")
    daily = daily.sort_index().copy()

    outcomes: list[float] = []
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    i = 80

    while i < len(h1) - 2:
        decision_time = h1.index[i]
        h1_hist = h1.iloc[: i + 1]
        h4_hist = h4.loc[:decision_time]
        daily_hist = daily.loc[:decision_time]
        if len(h4_hist) < 30 or len(daily_hist) < 60:
            i += 1
            continue

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
        risk = abs(entry - stop)
        if risk <= 0:
            i += 1
            continue

        search_end = min(i + 1 + entry_valid_bars, len(h1))
        entry_start = i + 1
        trigger_idx = None
        for j in range(entry_start, search_end):
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

        future = h1.iloc[trigger_idx : min(trigger_idx + max_hold_bars, len(h1))]
        r = _trade_result(plan, future, entry, stop, target)
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
    symbols = {"USDJPY": "JPY=X", "GOLD_FUTURES_GC": "GC=F"}
    results = []
    for name, symbol in symbols.items():
        daily, h1 = load_data(symbol)
        results.append(asdict(evaluate(name, daily, h1)))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
