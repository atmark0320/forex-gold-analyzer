from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtest_strategy import load_data
from strategy_candidates import ADX_GATES, CANDIDATES, FULL_ALIGNMENT, RISING_ADX, evaluate_candidate
from strategy_profiles import production_profile, save_active_profile

WINDOW_DAYS = 180
STEP_DAYS = 90
WINDOWS = 7
MIN_TRADES = 20
HEALTH_FILE = Path("production_health.json")
OUTPUT = Path("auto_profile_recommendation.json")
AUTO_SWITCH_ENV = "AUTO_PROFILE_SWITCH"


def _candidate_names(symbol: str) -> list[str]:
    """Return every candidate that can be evaluated for the symbol."""
    return list(CANDIDATES)


def _rolling_summary(symbol: str, variant: str, h1: pd.DataFrame):
    if variant not in CANDIDATES:
        raise ValueError(f"Unsupported candidate {variant} for {symbol}")
    end = h1.index.max()
    rows = []
    for window in range(1, WINDOWS + 1):
        window_end = end - pd.Timedelta(days=STEP_DAYS * (window - 1))
        window_start = window_end - pd.Timedelta(days=WINDOW_DAYS)
        result = evaluate_candidate(
            symbol,
            h1,
            cfg=CANDIDATES[variant],
            adx_gate=ADX_GATES.get(variant),
            full_alignment=variant in FULL_ALIGNMENT,
            rising_adx=variant in RISING_ADX,
            trade_start=window_start,
            trade_end=window_end + pd.Timedelta(hours=1),
        )
        rows.append({
            "symbol": symbol, "variant": variant, "window": window,
            "window_start": window_start.isoformat(), "window_end": window_end.isoformat(),
            "trades": int(result.trades), "wins": int(result.wins), "losses": int(result.losses),
            "total_r": float(result.total_r), "expectancy_r": float(result.expectancy_r),
            "profit_factor": float(result.profit_factor), "max_drawdown_r": float(result.max_drawdown_r),
            "win_rate_pct": float(result.win_rate_pct),
        })
    frame = pd.DataFrame(rows)
    summary = {
        "symbol": symbol, "variant": variant, "windows": int(len(frame)),
        "positive_windows": int((frame["total_r"] > 0).sum()),
        "negative_windows": int((frame["total_r"] < 0).sum()),
        "total_r": round(float(frame["total_r"].sum()), 3),
        "median_window_r": round(float(frame["total_r"].median()), 3),
        "worst_window_r": round(float(frame["total_r"].min()), 3),
        "best_window_r": round(float(frame["total_r"].max()), 3),
        "total_trades": int(frame["trades"].sum()),
    }
    return summary, rows


def _degraded_symbols() -> list[str]:
    try:
        payload = json.loads(HEALTH_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [str(row.get("symbol", "")).upper() for row in payload.get("symbols", [])
            if str(row.get("status", "")).lower() in {"degraded", "weak"}]


def _select_replacement(symbol: str, h1: pd.DataFrame):
    current = production_profile(symbol)
    if current not in CANDIDATES:
        current = "baseline"
    current_summary, current_rows = _rolling_summary(symbol, current, h1)
    evaluated = []
    best = None
    for variant in _candidate_names(symbol):
        summary, rows = _rolling_summary(symbol, variant, h1)
        evaluated.append(summary)
        if variant == current or summary["total_trades"] < MIN_TRADES:
            continue
        # Require a meaningful improvement in aggregate and median performance,
        # while preventing a materially worse worst window.
        if summary["total_r"] <= current_summary["total_r"] + 2.0:
            continue
        if summary["median_window_r"] <= current_summary["median_window_r"] + 0.5:
            continue
        if summary["positive_windows"] < max(4, WINDOWS // 2):
            continue
        if summary["worst_window_r"] < current_summary["worst_window_r"] - 2.0:
            continue
        score = summary["total_r"] + summary["median_window_r"] * 1.5 + summary["best_window_r"] * 0.5
        candidate = {
            "symbol": symbol, "current_profile": current, "recommended_profile": variant,
            "score": round(float(score), 3),
            "reason": f"総合成績 {summary['total_r']}R / median {summary['median_window_r']}R / positive_windows {summary['positive_windows']}/{summary['windows']}",
            "current_summary": current_summary, "candidate_summary": summary, "rows": rows,
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    return best, current_summary, evaluated


def main() -> None:
    recommendations = []
    for symbol in _degraded_symbols():
        h1 = load_data(symbol)
        candidate, current_summary, evaluated = _select_replacement(symbol, h1)
        if candidate is None:
            recommendations.append({
                "symbol": symbol, "status": "no_safe_replacement",
                "current_profile": production_profile(symbol),
                "reason": "全候補を評価したが、安全条件を満たす候補なし",
                "current_summary": current_summary, "evaluated_candidates": evaluated,
            })
        else:
            recommendations.append({**{k: candidate[k] for k in ("symbol", "current_profile", "recommended_profile", "reason", "score", "current_summary", "candidate_summary")},
                                    "status": "recommended", "evaluated_candidates": evaluated})

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "auto_profile_switch": os.environ.get(AUTO_SWITCH_ENV, "0").strip() == "1",
        "candidate_count": len(CANDIDATES), "recommendations": recommendations,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if payload["auto_profile_switch"]:
        for item in recommendations:
            if item.get("status") == "recommended":
                save_active_profile(item["symbol"], item["recommended_profile"], reason=item["reason"])
    for item in recommendations:
        print(f"{item['symbol']}: {item.get('status')} / {item.get('current_profile', '-')} -> {item.get('recommended_profile', '-')}")


if __name__ == "__main__":
    main()
