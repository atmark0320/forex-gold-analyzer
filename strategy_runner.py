from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pandas as pd

from market_data_provider import build_timeframes, fetch_ohlc
from strategy_profiles import production_adx_gate, production_config, production_profile
from technical_engine import generate_trade_plan

SYMBOLS = ("USDJPY", "XAUUSD")


def expected_4h_start(now: datetime) -> datetime:
    """Return the start of the most recently completed UTC 4H block."""
    current = now.astimezone(timezone.utc)
    block_hour = (current.hour // 4) * 4
    start = current.replace(hour=block_hour, minute=0, second=0, microsecond=0)
    # The current block is still forming, including exactly at its boundary.
    return start - pd.Timedelta(hours=4)


def _fmt(value: float | None, digits: int = 5) -> str:
    if value is None or not pd.notna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def _get_adx(h4: pd.DataFrame) -> float:
    """Read the canonical ADX column with an explicit validation error."""
    if "adx" not in h4.columns:
        raise RuntimeError(
            f"ADX column missing from completed H4 timeframe: {list(h4.columns)}"
        )
    if h4.empty:
        raise RuntimeError("completed H4 timeframe is empty")
    value = h4["adx"].iloc[-1]
    if pd.isna(value):
        raise RuntimeError("latest completed H4 ADX is NaN")
    return float(value)


def _apply_profile_gate(symbol: str, plan, h4: pd.DataFrame):
    gate = production_adx_gate(symbol)
    if gate is None:
        return plan

    adx = _get_adx(h4)
    if adx >= gate:
        return plan

    return replace(
        plan,
        direction="wait",
        confidence="low",
        entry_type="wait",
        entry_price=None,
        stop_price=None,
        target1=None,
        target2=None,
        risk_per_unit=None,
        reward_r1=None,
        reward_r2=None,
        reasons=tuple(
            list(plan.reasons)
            + [f"4H ADX {adx:.2f} < {gate:.0f} のため見送り"]
        ),
        invalidation=f"4H ADXが{gate:.0f}以上になるまで見送り",
    )


def build_production_plan(symbol: str, h1: pd.DataFrame):
    daily, h4 = build_timeframes(h1)
    if daily.empty or h4.empty or h1.empty:
        raise RuntimeError("insufficient completed timeframe data")
    cfg = production_config(symbol)
    plan = generate_trade_plan(daily, h4, h1, cfg)
    return _apply_profile_gate(symbol, plan, h4), {
        "D1": daily,
        "H4": h4,
        "H1": h1,
    }


def build_report(symbol: str, mtf: dict[str, pd.DataFrame], plan) -> str:
    d1 = mtf["D1"].iloc[-1]
    h4 = mtf["H4"].iloc[-1]
    h1 = mtf["H1"].iloc[-1]
    market_name = "XAU/USD spot" if symbol == "XAUUSD" else "USD/JPY spot"
    lines = [
        f"## {symbol} / {market_name}",
        f"Strategy profile: {production_profile(symbol)}",
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
        f"D1: {d1.get('dow_trend', '-')}",
        f"4H: {h4.get('dow_trend', '-')}",
        f"1H: {h1.get('dow_trend', '-')}",
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
        "※本番条件: 数値判断は決定論的なStrategyConfigで生成。XAUUSDの代替プロファイルは環境変数FOREX_XAU_PROFILEで明示指定した場合のみ有効。",
        "※本番監視対象はUSDJPY/XAUUSD。その他5銘柄は検証専用。",
        "※H1/H4/D1は確定足のみを使用。",
    ])
    return "\n".join(lines)


def main() -> None:
    for symbol in SYMBOLS:
        try:
            h1 = fetch_ohlc(symbol, days=450, end=datetime.now(timezone.utc))
            plan, mtf = build_production_plan(symbol, h1)
            print(build_report(symbol, mtf, plan))
        except Exception as exc:
            print(f"{symbol}: ERROR: {exc}")
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
