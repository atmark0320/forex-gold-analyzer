"""Walk-forward evaluator for the deterministic strategy engine.

No future bars are used to build a signal. Entry is evaluated on the next bar,
and TP/SL are evaluated from subsequent OHLC bars. When TP and SL occur in the
same bar, the conservative assumption is that SL was hit first.
"""

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
    max_drawdown_r: float


def load_1h(symbol: str) -> pd.DataFrame:
    df = yf.Ticker(symbol).history(period="60d", interval="1h", auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No data for {symbol}")
    return df


def evaluate(symbol: str, df: pd.DataFrame, cfg: StrategyConfig = StrategyConfig()) -> BacktestResult:
    h1 = df.sort_index().copy()
    h4 = resample_ohlc(h1, "4h")
    # For a short validation window, build daily bars from the same 1H history.
    # Production uses the longer daily history in strategy_runner.py.
    daily = resample_ohlc(h1, "1D")

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    outcomes: list[float] = []

    # Use only completed bars before the decision bar.
    for i in range(80, len(h1) - 2):
        decision_time = h1.index[i]
        h1_hist = h1.iloc[: i + 1]
        h4_hist = h4.loc[:decision_time]
        daily_hist = daily.loc[:decision_time]
        if len(h4_hist) < 30 or len(daily_hist) < 20:
            continue
        try:
            plan = generate_trade_plan(daily_hist, h4_hist, h1_hist, cfg)
        except Exception:
            continue
        if plan.direction == "wait" or plan.entry_price is None:
            continue

        # Next bar is the earliest possible execution; no same-bar hindsight.
        entry_bar = h1.iloc[i + 1]
        entry = float(plan.entry_price)
        stop = float(plan.stop_price)
        target = float(plan.target1)
        risk = abs(entry - stop)
        if risk <= 0:
            continue

        if plan.direction == "buy":
            if float(entry_bar["High"]) < entry:
                continue
            for _, bar in h1.iloc[i + 1 :].iterrows():
                hit_sl = float(bar["Low"]) <= stop
                hit_tp = float(bar["High"]) >= target
                if hit_sl and hit_tp:
                    r = -1.0
                    break
                if hit_sl:
                    r = -1.0
                    break
                if hit_tp:
                    r = abs(target - entry) / risk
                    break
            else:
                continue
        else:
            if float(entry_bar["Low"]) > entry:
                continue
            for _, bar in h1.iloc[i + 1 :].iterrows():
                hit_sl = float(bar["High"]) >= stop
                hit_tp = float(bar["Low"]) <= target
                if hit_sl and hit_tp:
                    r = -1.0
                    break
                if hit_sl:
                    r = -1.0
                    break
                if hit_tp:
                    r = abs(entry - target) / risk
                    break
            else:
                continue

        outcomes.append(float(r))
        equity += float(r)
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    wins = sum(x > 0 for x in outcomes)
    losses = sum(x < 0 for x in outcomes)
    trades = len(outcomes)
    return BacktestResult(
        symbol=symbol,
        trades=trades,
        wins=wins,
        losses=losses,
        win_rate_pct=round(100 * wins / trades, 2) if trades else 0.0,
        total_r=round(sum(outcomes), 3),
        expectancy_r=round(sum(outcomes) / trades, 4) if trades else 0.0,
        max_drawdown_r=round(max_dd, 3),
    )


def main() -> None:
    symbols = {"USDJPY": "JPY=X", "GOLD": "GC=F"}
    results = []
    for name, symbol in symbols.items():
        results.append(asdict(evaluate(name, load_1h(symbol))))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
