from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime, timezone

import pandas as pd

from market_data_provider import get_multi_timeframe_data
from technical_engine import StrategyConfig, generate_trade_plan


SYMBOLS = ("USDJPY", "XAUUSD")

# Production follows the canonical deterministic StrategyConfig.
# Experimental OOS overlays are kept in strategy_candidates.py and are not
# promoted here unless they demonstrate stable rolling out-of-sample results.
PRODUCTION_CONFIG = StrategyConfig()


def _latest_completed_4h_block(now: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(now).tz_convert("UTC")
    return ts.floor("4h") - pd.Timedelta(hours=4)


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or not pd.notna(value):
        return "-"
    return f"{value:.{digits}f}"


def build_production_plan(mtf: dict[str, pd.DataFrame]):
    """Generate the production trade plan without experimental filters."""
    return generate_trade_plan(mtf, PRODUCTION_CONFIG)


def build_report(symbol: str, mtf: dict[str, pd.DataFrame], plan) -> str:
    d1 = mtf["D1"]
    h4 = mtf["H4"]
    h1 = mtf["H1"]

    d1_last = d1.iloc[-1]
    h4_last = h4.iloc[-1]
    h1_last = h1.iloc[-1]

    lines = [
        f"## {symbol} / XAUUSDならXAU/USD spot",
        f"判定時刻(UTC): {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"### 判定: {plan.direction}",
        f"Confidence: {plan.confidence} / Score: {plan.score}",
        f"Regime: {plan.regime}",
        f"Entry type: {plan.entry_type}",
        "",
        f"Entry: {_fmt(plan.entry_price)}",
        f"SL: {_fmt(plan.stop_price)}",
        f"TP1: {_fmt(plan.target1)}",
        f"TP2: {_fmt(plan.target2)}",
        f"Risk/unit: {_fmt(plan.risk_per_unit)}",
        f"R1: {_fmt(plan.reward_r1)} / R2: {_fmt(plan.reward_r2)}",
        "",
        "### Dow structure",
        f"D1: {d1_last.get('dow_structure', '-')}",
        f"4H: {h4_last.get('dow_structure', '-')}",
        f"1H: {h1_last.get('dow_structure', '-')}",
        f"4H swing high: {_fmt(h4_last.get('last_swing_high'))}",
        f"4H swing low: {_fmt(h4_last.get('last_swing_low'))}",
        "",
        "### Indicators",
        f"4H ADX: {_fmt(h4_last.get('adx'))}",
        f"1H RSI: {_fmt(h1_last.get('rsi'))}",
        f"1H MACD: {_fmt(h1_last.get('macd'))}",
        "",
        "### Reasons",
    ]

    lines.extend(f"- {reason}" for reason in plan.reasons)
    lines.extend([
        "",
        f"Invalidation: {plan.invalidation}",
        "",
        "※本番条件: 実験的なADXゲート・高速TP・上昇ADX等は本番判定に使用せず、StrategyConfigの基準ロジックを使用。",
        "※4H/H1/D1はいずれも確定足のみを使用。",
    ])
    return "\n".join(lines)


def main() -> None:
    now = datetime.now(timezone.utc)
    completed_4h = _latest_completed_4h_block(now)

    for symbol in SYMBOLS:
        mtf = get_multi_timeframe_data(symbol)
        if not all(len(mtf.get(tf, pd.DataFrame())) for tf in ("D1", "H4", "H1")):
            print(f"{symbol}: insufficient completed timeframe data")
            continue

        latest_h4 = pd.Timestamp(mtf["H4"].index[-1]).tz_convert("UTC")
        if latest_h4 > completed_4h:
            print(f"{symbol}: latest H4 {latest_h4} is beyond expected completed block {completed_4h}")

        plan = build_production_plan(mtf)
        print(build_report(symbol, mtf, plan))
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
