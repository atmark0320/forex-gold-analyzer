"""All-symbol strategy candidate screening on the same out-of-sample window.

The baseline remains unchanged. Candidate overlays are test-only and every
result is retained. Promotion requires evidence across all seven symbols and
both IS/OOS validation.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import backtest_strategy
from backtest_strategy import DEFAULT_SYMBOLS, evaluate, load_data
from technical_engine import StrategyConfig, TradePlan

BASE = StrategyConfig()
CANDIDATES = {
    "baseline": BASE,
    "stricter_confirmation": replace(BASE, min_score=8, strong_score=10),
    "wider_target": replace(BASE, target1_r=1.8, target2_r=2.8),
    "faster_target": replace(BASE, target1_r=1.25, target2_r=2.0),
    "wider_stop": replace(BASE, stop_atr=1.60),
    "larger_entry_buffer": replace(BASE, entry_buffer_atr=0.15),
    "stricter_faster_target": replace(BASE, min_score=8, strong_score=10, target1_r=1.25, target2_r=2.0),
    "stricter_larger_buffer": replace(BASE, min_score=8, strong_score=10, entry_buffer_atr=0.15),
    "faster_larger_buffer": replace(BASE, target1_r=1.25, target2_r=2.0, entry_buffer_atr=0.15),
    "stricter_faster_larger_buffer": replace(BASE, min_score=8, strong_score=10, target1_r=1.25, target2_r=2.0, entry_buffer_atr=0.15),
    "adx25_gate": BASE,
    "adx22_gate": BASE,
    "adx25_faster_target": replace(BASE, target1_r=1.25, target2_r=2.0),
    "full_alignment": BASE,
    "full_alignment_faster_target": replace(BASE, target1_r=1.25, target2_r=2.0),
    "full_alignment_adx25": BASE,
}

ADX_GATES = {
    "adx25_gate": 25.0,
    "adx22_gate": 22.0,
    "adx25_faster_target": 25.0,
    "full_alignment_adx25": 25.0,
}

FULL_ALIGNMENT = {
    "full_alignment",
    "full_alignment_faster_target",
    "full_alignment_adx25",
}


def oos_days() -> int:
    try:
        return max(30, int(os.environ.get("OOS_DAYS", "180")))
    except ValueError:
        return 180


def evaluate_candidate(symbol, h1, cfg, adx_gate=None, full_alignment=False, **kwargs):
    """Run a test-only overlay without changing production strategy logic."""
    if adx_gate is None and not full_alignment:
        return evaluate(symbol, h1, cfg=cfg, **kwargs)

    original = backtest_strategy._prepared_plan

    def gated_plan(daily, h4, h1_snapshot, row, candidate_cfg):
        plan = original(daily, h4, h1_snapshot, row, candidate_cfg)
        if full_alignment:
            aligned = daily["dow_trend"] == h4["dow_trend"] == h1_snapshot["dow_trend"]
            if not aligned or daily["dow_trend"] not in {"up", "down"}:
                return replace(
                    plan,
                    direction="wait", confidence="low", entry_type="wait",
                    entry_price=None, stop_price=None, target1=None, target2=None,
                    risk_per_unit=None, reward_r1=None, reward_r2=None,
                    reasons=tuple(list(plan.reasons) + ["日足・4H・1Hのダウ方向が完全一致しないため見送り"]),
                    invalidation="3時間足すべてで同一方向のダウ構造が確認されるまで見送り",
                )
        if adx_gate is not None and float(h4["adx"]) < adx_gate:
            return replace(
                plan,
                direction="wait", confidence="low", entry_type="wait",
                entry_price=None, stop_price=None, target1=None, target2=None,
                risk_per_unit=None, reward_r1=None, reward_r2=None,
                reasons=tuple(list(plan.reasons) + [f"4H ADX {float(h4['adx']):.2f} < {adx_gate:.0f} のため見送り"]),
                invalidation=f"4H ADXが{adx_gate:.0f}以上になるまで見送り",
            )
        return plan

    backtest_strategy._prepared_plan = gated_plan
    try:
        return evaluate(symbol, h1, cfg=cfg, **kwargs)
    finally:
        backtest_strategy._prepared_plan = original


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
            result = evaluate_candidate(
                symbol, h1, cfg=cfg,
                adx_gate=ADX_GATES.get(name),
                full_alignment=name in FULL_ALIGNMENT,
                trade_start=start,
                trade_end=end + __import__("pandas").Timedelta(hours=1),
            )
            row = asdict(result)
            row.update({
                "variant": name,
                "evaluation": "out_of_sample",
                "oos_days": days,
                "adx_gate": ADX_GATES.get(name),
                "full_alignment": name in FULL_ALIGNMENT,
            })
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
