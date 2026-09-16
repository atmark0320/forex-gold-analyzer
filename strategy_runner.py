from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from market_data_provider import build_timeframes, fetch_ohlc
from technical_engine import StrategyConfig, generate_trade_plan

SYMBOLS = ("USDJPY", "XAUUSD")
PRODUCTION_CONFIG = StrategyConfig()


def _fmt(value: float | None, digits: int = 5) -> str:
    if value is None or not pd.notna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def build_production_plan(h1: pd.DataFrame):
    daily, h4 = build_timeframes(h1)
    if daily.empty or h4.empty or h1.empty:
        raise RuntimeError("insufficient completed timeframe data")
    # generate_trade_plan consumes completed D1/H4/H1 data. H1 is already
    # stripped of the forming candle by fetch_ohlc().
    mtf = {"D1": daily, "H4": h4, "H1": h1}
    return generate_trade_plan(mtf, PRODUCTION_CONFIG), mtf


def build_report(symbol: str, mtf: dict[str, pd.DataFrame], plan) -> str:
    d1 = mtf["D1"].iloc[-1]
    h4 = mtf["H4"].iloc[-1]
    h1 = mtf["H1"].iloc[-1]
    lines = [
        f"## {symbol} / XAUUSDならXAU/USD spot",
        f"判定時刻(UTC): {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"判定: {plan.direction.upper()}",
        f"Confidence: {plan.confidence} / Score: {plan.score}",
        f"Regime: {plan.regime}",
        f"Entry type: {plan.entry_type}",
        f"Entry: {_fmt(plan.entry_price)}",
        f"SL: {_fmt(plan.stop_price)}",
        f"TP1: {_fmt(plan.target1)}",
        f"TP2: {_fmt(plan.target2)}",
        f"Risk/unit: {_fmt(plan.risk_per_unit)}",
        f"R1: {_fmt(plan.reward_r1, 2)} / R2: {_fmt(plan.reward_r2, 2)}",
        "",
        "### Dow structure",
        f"D1: {d1.get('dow_structure', '-')}",
        f"4H: {h4.get('dow_structure', '-')}",
        f"1H: {h1.get('dow_structure', '-')}",
        f"4H swing high: {_fmt(h4.get('last_swing_high'))}",
        f"4H swing low: {_fmt(h4.get('last_swing_low'))}",
        "",
        "### Indicators",
        f"4H ADX: {_fmt(h4.get('adx'), 2)}",
        f"1H RSI: {_fmt(h1.get('rsi'), 2)}",
        f"1H MACD: {_fmt(h1.get('macd'), 5)}",
        "",
        "### Reasons",
    ]
    lines.extend(f"- {reason}" for reason in plan.reasons)
    lines.extend([
        "",
        f"Invalidation: {plan.invalidation}",
        "",
        "※本番条件: StrategyConfigの基準ロジックを使用。実験的なADXゲート・高速TP・上昇ADX条件は本番判定に使用しない。",
        "※H1/H4/D1は確定足のみを使用。",
    ])
    return "\n".join(lines)


def main() -> None:
    # Fetching through the existing provider preserves the Dukascopy spot
    # source/fallback and its forming-candle protection.
    for symbol in SYMBOLS:
        try:
            h1 = fetch_ohlc(symbol, days=450, end=datetime.now(timezone.utc))
            plan, mtf = build_production_plan(h1)
            print(build_report(symbol, mtf, plan))
        except Exception as exc:
            print(f"{symbol}: ERROR: {exc}")
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
