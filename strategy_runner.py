"""Production entrypoint for deterministic USD/JPY and Gold strategies."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

from technical_engine import StrategyConfig, generate_trade_plan, resample_ohlc, timeframe_snapshot

JST = timezone(timedelta(hours=9))
RUN_TS = datetime.now(JST).strftime("%Y%m%d_%H%M")
PAYLOAD_FILE = "notify_payloads.json"

SYMBOLS = {
    "ドル円 (USD/JPY)": "JPY=X",
    "金 (GOLD / GC=F)": "GC=F",
}


def fetch(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return long-history daily data and recent 1H data."""
    ticker = yf.Ticker(symbol)
    daily = ticker.history(period="2y", interval="1d", auto_adjust=False)
    h1 = ticker.history(period="60d", interval="1h", auto_adjust=False)
    if daily.empty or h1.empty:
        raise RuntimeError(f"{symbol}: market data is empty")
    return daily, h1


def fmt_price(value: float, symbol: str) -> str:
    digits = 3 if symbol == "JPY=X" else 2
    return f"{value:.{digits}f}"


def dow_name(value: str) -> str:
    return {
        "up": "上昇（高値・安値切り上げ）",
        "down": "下降（高値・安値切り下げ）",
        "range": "レンジ",
        "transition": "転換途中",
    }.get(value, value)


def build_report(name: str, symbol: str, daily: pd.DataFrame, h1: pd.DataFrame) -> str:
    cfg = StrategyConfig()
    h4 = resample_ohlc(h1, "4h")
    plan = generate_trade_plan(daily, h4, h1, cfg)
    ds, hs4, hs1 = timeframe_snapshot(daily, cfg), timeframe_snapshot(h4, cfg), timeframe_snapshot(h1, cfg)
    p = plan.to_dict()

    direction = {"buy": "買い", "sell": "売り", "wait": "見送り"}[p["direction"]]
    lines = [
        f"【{name}】ダウ理論ベース・テクニカル戦略 ({RUN_TS} JST)",
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
        "※ダウ理論を予測の根幹とし、EMA・RSI・MACD・ADXは補助的な確認に使用しています。",
        "※これはテクニカル分析による戦略候補であり、利益や将来価格を保証する予測ではありません。",
    ]
    return "\n".join(lines)


def main() -> None:
    payloads: list[dict] = []
    failures: list[str] = []
    for name, symbol in SYMBOLS.items():
        try:
            daily, h1 = fetch(symbol)
            text = build_report(name, symbol, daily, h1)
            payloads.append({"type": "text", "text": text})
            print(text)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"ERROR {name}: {exc}")

    with open(PAYLOAD_FILE, "w", encoding="utf-8") as f:
        json.dump(payloads, f, ensure_ascii=False, indent=2)
    if not payloads:
        raise RuntimeError("USD/JPY・Goldの両方で戦略生成に失敗しました: " + "; ".join(failures))


if __name__ == "__main__":
    main()
