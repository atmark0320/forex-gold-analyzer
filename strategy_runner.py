"""Production entrypoint for deterministic USD/JPY and XAU/USD spot strategies."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from market_data_provider import build_timeframes, fetch_ohlc, latest_completed_4h_time
from technical_engine import StrategyConfig, generate_trade_plan, timeframe_snapshot

JST = timezone(timedelta(hours=9))
UTC = timezone.utc
RUN_TS = datetime.now(JST).strftime("%Y%m%d_%H%M")
PAYLOAD_FILE = "notify_payloads.json"

SYMBOLS = {
    "ドル円 (USD/JPY)": "USDJPY",
    "金 (XAU/USD スポット)": "XAUUSD",
}

# Production policy promoted from the 180-day OOS candidate screening:
# 1) require 4H ADX >= 25 to avoid weak/non-trending entries
# 2) use faster partial/primary targets (1.25R / 2.0R)
# The underlying Dow/indicator calculation remains unchanged.
PRODUCTION_CONFIG = replace(
    StrategyConfig(),
    target1_r=1.25,
    target2_r=2.0,
)
PRODUCTION_MIN_ADX = 25.0


def fmt_price(value: float, symbol: str) -> str:
    digits = 3 if symbol == "USDJPY" else 2
    return f"{value:.{digits}f}"


def dow_name(value: str) -> str:
    return {
        "up": "上昇（高値・安値切り上げ）",
        "down": "下降（高値・安値切り下げ）",
        "range": "レンジ",
        "transition": "転換途中",
    }.get(value, value)


def expected_4h_start(now: datetime | None = None):
    current = now or datetime.now(UTC)
    stamp = datetime.fromtimestamp(current.timestamp(), tz=UTC)
    block_hour = (stamp.hour // 4) * 4
    current_block = stamp.replace(hour=block_hour, minute=0, second=0, microsecond=0)
    return current_block - timedelta(hours=4)


def is_new_4h_ready(h1, now: datetime | None = None) -> tuple[bool, str]:
    latest = latest_completed_4h_time(h1)
    expected = expected_4h_start(now)
    if latest is None:
        return False, "確定済み4H足がありません"
    latest_utc = latest.to_pydatetime().astimezone(UTC)
    if latest_utc >= expected:
        return True, f"分析対象4H足: {latest_utc.isoformat()}"
    return False, f"最新の完全4H足 {latest_utc.isoformat()} / 今回対象 {expected.isoformat()}"


def build_production_plan(daily, h4, h1):
    """Generate the live plan using the OOS-screened production overlay."""
    plan = generate_trade_plan(daily, h4, h1, PRODUCTION_CONFIG)
    hs4 = timeframe_snapshot(h4, PRODUCTION_CONFIG)
    if float(hs4["adx"]) < PRODUCTION_MIN_ADX:
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
                + [f"4H ADX {float(hs4['adx']):.2f} < {PRODUCTION_MIN_ADX:.0f} のため見送り"]
            ),
            invalidation=f"4H ADXが{PRODUCTION_MIN_ADX:.0f}以上になるまで見送り",
        )
    return plan


def build_report(name: str, symbol: str, daily, h4, h1, h4_start) -> str:
    cfg = PRODUCTION_CONFIG
    plan = build_production_plan(daily, h4, h1)
    ds = timeframe_snapshot(daily, cfg)
    hs4 = timeframe_snapshot(h4, cfg)
    hs1 = timeframe_snapshot(h1, cfg)
    p = plan.to_dict()

    direction = {"buy": "買い", "sell": "売り", "wait": "見送り"}[p["direction"]]
    h4_label = h4_start.astimezone(JST).strftime("%Y-%m-%d %H:%M JST開始")
    lines = [
        f"【{name}】4H確定時・ダウ理論ベース戦略 ({RUN_TS} JST)",
        f"分析4H足: {h4_label}",
        f"現在値: {fmt_price(hs1['close'], symbol)}",
        "",
        "■ダウ理論による環境認識（最重要）",
        f"日足: {dow_name(ds['dow_trend'])}",
        f"4H: {dow_name(hs4['dow_trend'])}",
        f"1H: {dow_name(hs1['dow_trend'])}",
        f"4H直近スイング高値: {fmt_price(hs4['last_swing_high'], symbol) if hs4['last_swing_high'] is not None else '未確定'}",
        f"4H直近スイング安値: {fmt_price(hs4['last_swing_low'], symbol) if hs4['last_swing_low'] is not None else '未確定'}",
        "",
        "■補助指標による確認",
        f"4H ADX: {hs4['adx']:.1f} / 1H RSI: {hs1['rsi']:.1f}",
        f"1H MACDヒストグラム: {hs1['macd_hist']:.5f}",
        "",
        "■最終判断",
        f"方向: {direction}",
        f"信頼度: {p['confidence']} / スコア: {p['score']}",
    ]
    if p["entry_price"] is not None:
        lines += [
            f"エントリー: {fmt_price(p['entry_price'], symbol)} ({'条件付き' if p['entry_type'] == 'conditional' else '即時候補'})",
            f"損切り: {fmt_price(p['stop_price'], symbol)}",
            f"目標1: {fmt_price(p['target1'], symbol)} ({p['reward_r1']}R)",
            f"目標2: {fmt_price(p['target2'], symbol)} ({p['reward_r2']}R)",
        ]
    else:
        lines.append("エントリー条件: 現時点では優位性不足。ダウ構造が確定するまで待機")
    lines += [
        "",
        "■根拠",
        *[f"・{reason}" for reason in p["reasons"]],
        "",
        f"■無効化条件: {p['invalidation']}",
        "※予測の根幹はダウ理論。EMA・RSI・MACD・ADXは補助的な確認に使用しています。",
        "※本番条件: 4H ADX 25未満は見送り、目標1/目標2は1.25R/2.0R。",
        "※XAU/USDはスポット価格（Dukascopy BID 1H）を使用しています。",
        "※これはテクニカル分析による戦略候補であり、利益や将来価格を保証するものではありません。",
    ]
    return "\n".join(lines)


def main() -> None:
    payloads: list[dict] = []
    failures: list[str] = []
    reference_h4 = None

    for name, symbol in SYMBOLS.items():
        try:
            h1 = fetch_ohlc(symbol, days=450)
            ready, detail = is_new_4h_ready(h1)
            print(f"{name}: {detail}")
            if not ready:
                continue

            daily, h4 = build_timeframes(h1)
            h4_start = latest_completed_4h_time(h1)
            if h4_start is None:
                continue
            if reference_h4 is None or h4_start > reference_h4:
                reference_h4 = h4_start

            text = build_report(name, symbol, daily, h1, h1, h4_start)
            payloads.append({"type": "text", "text": text})
            print(text)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"ERROR {name}: {exc}")

    if not payloads and failures:
        raise RuntimeError("USD/JPY・XAU/USDスポットの戦略生成に失敗しました: " + "; ".join(failures))

    if not payloads:
        print("No new complete 4H candle is ready; skipping notification.")
        return

    with open(PAYLOAD_FILE, "w", encoding="utf-8") as f:
        json.dump(payloads, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
